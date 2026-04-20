using UnityEngine;
using UnityEngine.UI;
using ROS2;
using sensor_msgs.msg;
using RosImage = sensor_msgs.msg.Image;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Net.Sockets;
using System.Text;

/// <summary>
/// Unity 侧统一 SSVEP 显示器（解码 + 预训练）。
/// - 解码模式：接收 ROS 图像批次，按 ROS 指令开始闪烁。
/// - 预训练模式：接收 ROS 的提示/刺激/休息指令，仅负责显示并回传 trial_start UDP。
///
/// 重要说明：
/// 控制状态机始终在 ROS 侧；Unity 只负责渲染和发送时序触发。
/// </summary>
public class ROS2SSVEPStimulator2 : MonoBehaviour
{
    // --------------------------
    // Inspector：UI 引用
    // --------------------------
    public ROS2UnityComponent ros2UnityComponent;
    public RawImage[] imageUIs;  // expected length >= 8
    public UnityEngine.UI.Image[] frameUIs;
    public GameObject stimulusPanel;

    // --------------------------
    // Inspector：ROS 话题
    // --------------------------
    public string decodeImageTopic = "/image_seg";
    public string decodeCommandTopic = "/ssvep_decode_cmd";
    public string trainCommandTopic = "/ssvep_train_cmd";
    public bool useReliableQos = true;
    public bool useManualRefreshRate = false;
    [Min(1)] public int manualRefreshRateHz = 165;
    [Header("Decode Flash Colors")]
    public Color decodeFlashOnColor = new Color(1f, 1f, 1f, 1f);
    public Color decodeFlashOffColor = new Color(0.0f, 0.0f, 0.0f, 1.0f);
    [Header("Image Flash")]
    public bool flashImagesWithFrames = false;
    [Range(0f, 1f)] public float imageFlashOnAlpha = 1f;
    [Range(0f, 1f)] public float imageFlashOffAlpha = 0.2f;
    [Header("SSVEP Sequence Debug")]
    public bool enableSequenceLogging = true;
    public int logFirstNFrames = 120;
    public string sequenceLogFileName = "ssvep_sequence_log.csv";
    public string debugTargetIdsToLog = "1,2,3,4,5,6,7,8";

    // 可见目标对应频率：target_id 1..8 -> UI 索引 0..7
    public float[] ssvepFrequencies = { 8.684f, 9.706f, 11.0f, 11.786f, 12.692f, 13.75f, 15.0f, 18.333f };

    // --------------------------
    // Inspector：UDP 端点
    // --------------------------
    // 解码标记（字节触发）：例如 100+target 表示开始，200+target 表示结束
    public string decodeTriggerTargetIP = "127.0.0.1";
    public int decodeTriggerTargetPort = 9999;

    // 发给 ROS 控制器的解码确认
    public string decodeAckTargetIP = "127.0.0.1";
    public int decodeAckPort = 10000;

    // 发给 ROS 控制器的预训练触发文本
    public string trainTriggerTargetIP = "127.0.0.1";
    public int trainTriggerTargetPort = 10001;

    private readonly int[] targetIndices = { 0, 1, 2, 3, 4, 5, 6, 7 };
    // 解码图像包仍只提供 6 张动态图，映射到这些槽位。
    private readonly int[] decodeImageIndices = { 0, 1, 2, 4, 5, 6 };

    private UdpClient udpSender;
    private ROS2Node node;
    private ISubscription<RosImage> decodeSubscription;
    private ISubscription<RosImage> decodeCommandSubscription;
    private ISubscription<RosImage> trainSubscription;

    // 解码数据流队列
    private readonly Queue<byte[]> pendingImageData = new Queue<byte[]>();
    private readonly Queue<(int width, int height)> pendingImageSizes = new Queue<(int, int)>();
    private readonly Queue<string> pendingImageFrameIds = new Queue<string>();

    // 解码控制命令队列
    private readonly Queue<string> pendingDecodeCmdFrameIds = new Queue<string>();

    // 训练命令队列
    private readonly Queue<string> pendingTrainCmdFrameIds = new Queue<string>();

    private readonly object queueLock = new object();

    private List<Texture2D> receivedTextures = new List<Texture2D>(new Texture2D[6]);
    private readonly HashSet<int> receivedImgIndices = new HashSet<int>();
    private bool isBatchCompleted = false;
    private bool batchStartedByCommand = false;
    private int batchTrialId = -1;
    private int expectedBatchImageCount = 6;
    private int activeDecodeSlotCount = 6;
    private readonly HashSet<int> activeDecodeFrameIndices = new HashSet<int>();

    private Coroutine[] ssvepCoroutines;
    private Coroutine globalFrameTickCoroutine;
    private bool[][] ssvepFramePatterns;
    private int[] ssvepPatternLengths;
    private int globalFrameCounter = 0;
    private readonly StringBuilder sequenceLogBuffer = new StringBuilder(16384);
    private readonly Dictionary<int, int> loggedFramesByTarget = new Dictionary<int, int>();
    private readonly HashSet<int> debugTargetIdSet = new HashSet<int>();
    private int sequenceLogPendingRows = 0;
    private bool logAllTargets = true;

    private int currentTrialId = -1;
    private int currentTargetId = -1;

    private enum VisualMode
    {
        None,
        Decode,
        Pretrain,
    }

    private VisualMode visualMode = VisualMode.None;

    private Texture[] defaultImageTextures;
    private Texture2D redImageTexture;
    private Texture2D whiteImageTexture;
    private bool decodeStimStarted = false;

    void Start()
    {
        int detectedRefreshHz = Screen.currentResolution.refreshRate > 0
            ? Screen.currentResolution.refreshRate
            : 60;

        foreach(var res in Screen.resolutions)
        {
            int hz = (int)res.refreshRateRatio.value;
            if(hz > detectedRefreshHz) detectedRefreshHz = hz;
        }

        int configuredRefreshHz = useManualRefreshRate
            ? Mathf.Max(1, manualRefreshRateHz)
            : Mathf.Max(1, detectedRefreshHz);

        QualitySettings.vSyncCount = 1;
        Application.runInBackground = true;
        Application.targetFrameRate = configuredRefreshHz;

        Debug.Log(
            $"[Display] currentResolution={Screen.currentResolution.width}x{Screen.currentResolution.height}@" +
            $"{Screen.currentResolution.refreshRate}Hz, " +
            $"detectedRefreshHz={detectedRefreshHz}, " +
            $"useManualRefreshRate={useManualRefreshRate}, " +
            $"manualRefreshRateHz={manualRefreshRateHz}, " +
            $"configuredRefreshHz={configuredRefreshHz}, " +
            $"vSyncCount={QualitySettings.vSyncCount}, " +
            $"targetFrameRate={Application.targetFrameRate}, " +
            $"runInBackground={Application.runInBackground}"
        );

        if (stimulusPanel == null || imageUIs == null || frameUIs == null)
        {
            Debug.LogError("Assign imageUIs/frameUIs/stimulusPanel in Inspector.");
            enabled = false;
            return;
        }

        if (targetIndices.Length != ssvepFrequencies.Length)
        {
            Debug.LogError("ssvepFrequencies length must match visible target count (8).");
            enabled = false;
            return;
        }

        RebuildSequenceLogTargetSet();
        BuildSsvEpFramePatterns(configuredRefreshHz);

        stimulusPanel.SetActive(false);
        decodeStimStarted = false;
        SetAllFramesAlpha(0f);
        SetAllImageAlpha(1f);
        UpdateActiveDecodeSlotCount(6);
        CacheDefaultImageTextures();
        InitializePretrainSolidTextures();
        ResetDecodeBatchState(clearQueues: true);

        try
        {
            udpSender = new UdpClient();
            Debug.Log($"UDP sender initialized decodeTrigger={decodeTriggerTargetIP}:{decodeTriggerTargetPort}, decodeAck={decodeAckTargetIP}:{decodeAckPort}, trainTrigger={trainTriggerTargetIP}:{trainTriggerTargetPort}");
        }
        catch (Exception e)
        {
            Debug.LogError("Failed to initialize UDP sender: " + e.Message);
            udpSender = null;
        }

        if (ros2UnityComponent == null)
        {
            ros2UnityComponent = FindObjectOfType<ROS2UnityComponent>();
        }
        if (ros2UnityComponent == null)
        {
            Debug.LogError("ROS2UnityComponent not found.");
            enabled = false;
            return;
        }

        node = ros2UnityComponent.CreateNode("ssvep_stimulator_2");
        var qos = new QualityOfServiceProfile();
        qos.SetHistory(HistoryPolicy.QOS_POLICY_HISTORY_KEEP_LAST, 10);
        qos.SetReliability(
            useReliableQos
                ? ReliabilityPolicy.QOS_POLICY_RELIABILITY_RELIABLE
                : ReliabilityPolicy.QOS_POLICY_RELIABILITY_BEST_EFFORT
        );

        decodeSubscription = node.CreateSubscription<RosImage>(decodeImageTopic, OnDecodeImageReceived, qos);
        decodeCommandSubscription = node.CreateSubscription<RosImage>(decodeCommandTopic, OnDecodeCommandReceived, qos);
        trainSubscription = node.CreateSubscription<RosImage>(trainCommandTopic, OnTrainCommandReceived, qos);

        Debug.Log(
            $"Subscribed decode image topic={decodeImageTopic}, " +
            $"decode command topic={decodeCommandTopic}, train topic={trainCommandTopic}"
        );
    }

    // 解码回调：处理解码采集模式下的图像批次。
    void OnDecodeImageReceived(RosImage msg)
    {
        string frameId = msg.Header != null ? msg.Header.Frame_id : "";

        if (msg.Encoding != "bgr8")
        {
            return;
        }

        byte[] dataCopy = new byte[msg.Data.Length];
        Buffer.BlockCopy(msg.Data, 0, dataCopy, 0, msg.Data.Length);
        lock (queueLock)
        {
            pendingImageData.Enqueue(dataCopy);
            pendingImageSizes.Enqueue(((int)msg.Width, (int)msg.Height));
            pendingImageFrameIds.Enqueue(frameId);
        }
    }

    void OnDecodeCommandReceived(RosImage msg)
    {
        string frameId = msg.Header != null ? msg.Header.Frame_id : "";
        if (string.IsNullOrWhiteSpace(frameId))
        {
            return;
        }
        lock (queueLock)
        {
            pendingDecodeCmdFrameIds.Enqueue(frameId);
        }
    }

    // 训练回调：处理提示/刺激/休息命令帧。
    void OnTrainCommandReceived(RosImage msg)
    {
        string frameId = msg.Header != null ? msg.Header.Frame_id : "";
        if (string.IsNullOrWhiteSpace(frameId))
        {
            return;
        }
        lock (queueLock)
        {
            pendingTrainCmdFrameIds.Enqueue(frameId);
        }
    }

    void Update()
    {
        lock (queueLock)
        {
            while (pendingDecodeCmdFrameIds.Count > 0)
            {
                string frameId = pendingDecodeCmdFrameIds.Dequeue();
                HandleDecodeCommand(frameId);
            }

            while (pendingTrainCmdFrameIds.Count > 0)
            {
                string frameId = pendingTrainCmdFrameIds.Dequeue();
                HandleTrainCommand(frameId);
            }

            while (pendingImageData.Count > 0)
            {
                byte[] data = pendingImageData.Dequeue();
                var (w, h) = pendingImageSizes.Dequeue();
                string frameId = pendingImageFrameIds.Dequeue();
                HandleDecodeImagePacket(data, w, h, frameId);
            }
        }
    }

    // --------------------------------------
    // 解码模式图像包处理
    // --------------------------------------
    void HandleDecodeImagePacket(byte[] data, int width, int height, string frameId)
    {
        int trialId, imgIdx, targetId;
        if (!TryParseDecodeFrameMeta(frameId, out trialId, out imgIdx, out targetId))
        {
            return;
        }

        if (batchStartedByCommand)
        {
            if (batchTrialId < 0)
            {
                batchTrialId = trialId;
                currentTrialId = trialId;
            }
            if (trialId != batchTrialId)
            {
                return;
            }
            if (targetId > 0)
            {
                currentTargetId = targetId;
            }
        }

        if (currentTrialId < 0)
        {
            currentTrialId = trialId;
            currentTargetId = targetId;
        }

        if (!batchStartedByCommand && trialId != currentTrialId)
        {
            if (trialId > currentTrialId)
            {
                StopDecodeStimulationKeepVisuals(sendDecodeEndTrigger: true);
                ResetDecodeBatchState(clearQueues: false);
                currentTrialId = trialId;
                currentTargetId = targetId;
            }
            else
            {
                return;
            }
        }

        int requiredCount = batchStartedByCommand ? expectedBatchImageCount : 6;
        if (
            imgIdx < 0
            || imgIdx >= 6
            || imgIdx >= requiredCount
            || receivedImgIndices.Contains(imgIdx)
        )
        {
            return;
        }

        Texture2D tex = BgrToTexture(data, width, height);
        receivedTextures[imgIdx] = tex;
        receivedImgIndices.Add(imgIdx);

        if (!isBatchCompleted && receivedImgIndices.Count >= requiredCount)
        {
            isBatchCompleted = true;
            AssignTexturesToUI();
            PrepareDecodeVisuals();
        }
    }

    Texture2D BgrToTexture(byte[] data, int width, int height)
    {
        Texture2D tex = new Texture2D(width, height, TextureFormat.RGB24, false);
        int pixelCount = width * height;
        byte[] rgb = new byte[data.Length];
        for (int i = 0; i < pixelCount; i++)
        {
            int idx = i * 3;
            rgb[idx + 0] = data[idx + 2];
            rgb[idx + 1] = data[idx + 1];
            rgb[idx + 2] = data[idx + 0];
        }
        tex.LoadRawTextureData(rgb);
        tex.Apply();
        return tex;
    }

    void AssignTexturesToUI()
    {
        if (imageUIs == null || imageUIs.Length <= 7)
        {
            return;
        }

        for (int i = 0; i < decodeImageIndices.Length; i++)
        {
            int uiIndex = decodeImageIndices[i];
            Texture2D tex = (i < receivedTextures.Count) ? receivedTextures[i] : null;
            bool isActiveSlot = i < activeDecodeSlotCount;
            if (isActiveSlot && tex != null)
            {
                imageUIs[uiIndex].texture = tex;
                imageUIs[uiIndex].gameObject.SetActive(true);
            }
            else
            {
                imageUIs[uiIndex].texture = null;
                imageUIs[uiIndex].gameObject.SetActive(false);
            }
        }

        // 解码阶段保持默认静态图标（勾/叉）可见。
        imageUIs[3].gameObject.SetActive(true);
        imageUIs[7].gameObject.SetActive(true);
    }

    void StartDecodeStimulation()
    {
        visualMode = VisualMode.Decode;
        StopAllCoroutines();
        StopGlobalFrameTick();
        ResetGlobalFrameCounter();
        ResetSequenceLogSession();
        if (stimulusPanel != null) stimulusPanel.SetActive(true);
        SetAllFramesAlpha(0f);
        SetAllImageAlpha(1f);

        StartGlobalFrameTick();
        ssvepCoroutines = new Coroutine[targetIndices.Length];
        for (int i = 0; i < targetIndices.Length; i++)
        {
            int frameIdx = targetIndices[i];
            if (frameIdx < frameUIs.Length)
            {
                if (!IsDecodeFrameActive(frameIdx))
                {
                    continue;
                }
                ssvepCoroutines[i] = StartCoroutine(SSVEPFlash(frameIdx, i, false));
            }
        }

        decodeStimStarted = true;
        SendDecodeMarker(100 + Mathf.Max(1, currentTargetId));
    }

    // --------------------------------------
    // 解码/训练命令处理
    // --------------------------------------
    void HandleDecodeCommand(string frameId)
    {
        if (!TryParseCommand(frameId, out string cmd, out int trialId, out int targetId, out int count))
        {
            return;
        }

        if (cmd == "batch_start")
        {
            BeginDecodeBatch(trialId, targetId, count);
            return;
        }

        if (cmd == "batch_end")
        {
            if (batchStartedByCommand && trialId > 0 && batchTrialId > 0 && trialId != batchTrialId)
            {
                return;
            }
            if (!isBatchCompleted && receivedImgIndices.Count >= expectedBatchImageCount)
            {
                isBatchCompleted = true;
                AssignTexturesToUI();
                PrepareDecodeVisuals();
            }
            return;
        }

        if (cmd == "prepare")
        {
            if (trialId > 0) currentTrialId = trialId;
            if (targetId > 0) currentTargetId = targetId;
            PrepareDecodeVisuals();
            return;
        }

        if (cmd == "stim")
        {
            if (trialId > 0) currentTrialId = trialId;
            if (targetId > 0) currentTargetId = targetId;
            if (isBatchCompleted)
            {
                StartDecodeStimulation();
            }
            return;
        }

        if (cmd == "stop")
        {
            StopDecodeStimulationKeepVisuals(sendDecodeEndTrigger: true);
            return;
        }

        if (cmd == "done")
        {
            StopDecodeStimulationKeepVisuals(sendDecodeEndTrigger: false);
            return;
        }
    }

    void HandleTrainCommand(string frameId)
    {
        if (!TryParseCommand(frameId, out string cmd, out int trialId, out int targetId, out int _))
        {
            return;
        }

        if (cmd == "cue")
        {
            currentTrialId = trialId;
            currentTargetId = targetId;
            ShowCue(targetId);
            return;
        }

        if (cmd == "stim")
        {
            currentTrialId = trialId;
            currentTargetId = targetId;
            StartTrainStimulation();
            return;
        }

        if (cmd == "rest" || cmd == "stop" || cmd == "done")
        {
            StopCurrentStimulationVisuals(sendDecodeEndTrigger: false);
            RestoreDefaultImageTextures();
            return;
        }
    }

    void ShowCue(int targetId)
    {
        visualMode = VisualMode.Pretrain;
        StopAllCoroutines();
        if (stimulusPanel != null) stimulusPanel.SetActive(true);

        SetAllFramesAlpha(0.15f);
        SetAllImageAlpha(1f);
        ApplyPretrainTargetLayout(targetId);

        int frameIdx = TargetIdToFrameIndex(targetId);
        if (frameIdx >= 0 && frameIdx < frameUIs.Length)
        {
            frameUIs[frameIdx].color = new Color(1f, 0f, 0f, 1f);
        }
    }

    void StartTrainStimulation()
    {
        visualMode = VisualMode.Pretrain;
        StopAllCoroutines();
        StopGlobalFrameTick();
        ResetGlobalFrameCounter();
        ResetSequenceLogSession();
        if (stimulusPanel != null) stimulusPanel.SetActive(true);

        ApplyPretrainTargetLayout(currentTargetId);
        SetAllFramesAlpha(0f);
        SetAllImageAlpha(1f);
        StartGlobalFrameTick();
        ssvepCoroutines = new Coroutine[targetIndices.Length];
        for (int i = 0; i < targetIndices.Length; i++)
        {
            int frameIdx = targetIndices[i];
            if (frameIdx < frameUIs.Length)
            {
                bool isTarget = (i + 1) == currentTargetId;
                ssvepCoroutines[i] = StartCoroutine(SSVEPFlash(frameIdx, i, isTarget));
            }
        }

        SendTrainTrialStart();
    }

    IEnumerator SSVEPFlash(int frameUiIndex, int patternIndex, bool keepRedTarget)
    {
        while (true)
        {
            bool isOn = false;
            int patternLength = 0;
            int cursorBeforeAdvance = 0;
            if (TryGetPatternFrameState(patternIndex, globalFrameCounter, out bool patternOn, out int len, out int cursor))
            {
                isOn = patternOn;
                patternLength = len;
                cursorBeforeAdvance = cursor;
            }

            MaybeLogSequenceFrame(
                patternIndex + 1,
                patternLength,
                cursorBeforeAdvance,
                isOn
            );

            if (keepRedTarget)
            {
                frameUIs[frameUiIndex].color = isOn
                    ? new Color(1f, 0f, 0f, 1f)
                    : new Color(1f, 0f, 0f, 0.25f);
            }
            else
            {
                frameUIs[frameUiIndex].color = isOn
                    ? decodeFlashOnColor
                    : decodeFlashOffColor;
            }

            if (flashImagesWithFrames
                && imageUIs != null
                && frameUiIndex >= 0
                && frameUiIndex < imageUIs.Length
                && imageUIs[frameUiIndex] != null
                && imageUIs[frameUiIndex].gameObject.activeSelf)
            {
                Color baseColor = imageUIs[frameUiIndex].color;
                float alpha = isOn ? imageFlashOnAlpha : imageFlashOffAlpha;
                imageUIs[frameUiIndex].color = new Color(baseColor.r, baseColor.g, baseColor.b, alpha);
            }

            yield return null;
        }
    }

    IEnumerator GlobalFrameTick()
    {
        while (true)
        {
            yield return null;
            globalFrameCounter++;
        }
    }

    void StartGlobalFrameTick()
    {
        StopGlobalFrameTick();
        globalFrameTickCoroutine = StartCoroutine(GlobalFrameTick());
    }

    void StopGlobalFrameTick()
    {
        if (globalFrameTickCoroutine != null)
        {
            StopCoroutine(globalFrameTickCoroutine);
            globalFrameTickCoroutine = null;
        }
    }

    void BuildSsvEpFramePatterns(int refreshRateHz)
    {
        ssvepFramePatterns = new bool[ssvepFrequencies.Length][];
        ssvepPatternLengths = new int[ssvepFrequencies.Length];

        for (int i = 0; i < ssvepFrequencies.Length; i++)
        {
            float frequency = ssvepFrequencies[i];
            int sequenceLength = 1;
            if (frequency > 0f)
            {
                // Positive values only: +0.5 then floor gives nearest integer,
                // and .5 ties go upward (away from zero).
                sequenceLength = Mathf.Max(
                    1,
                    (int)Math.Floor((refreshRateHz / frequency) + 0.5f)
                );
            }

            int onCount = sequenceLength / 2;
            int offCount = sequenceLength - onCount;

            bool[] pattern = new bool[sequenceLength];
            for (int f = 0; f < sequenceLength; f++)
            {
                pattern[f] = f < onCount;
            }

            ssvepFramePatterns[i] = pattern;
            ssvepPatternLengths[i] = sequenceLength;

            Debug.Log(
                $"[SSVEPPattern] target={i + 1}, freq={frequency:F3}Hz, " +
                $"refresh={refreshRateHz}, length={sequenceLength}, on={onCount}, off={offCount}"
            );
        }
    }

    void ResetGlobalFrameCounter()
    {
        globalFrameCounter = 0;
        Debug.Log("[SSVEPPattern] globalFrameCounter reset to 0");
    }

    bool TryGetPatternFrameState(
        int patternIndex,
        int frameCounter,
        out bool isOn,
        out int patternLength,
        out int cursorBeforeAdvance
    )
    {
        isOn = false;
        patternLength = 0;
        cursorBeforeAdvance = 0;

        if (ssvepFramePatterns == null || ssvepPatternLengths == null)
        {
            return false;
        }
        if (patternIndex < 0 || patternIndex >= ssvepFramePatterns.Length)
        {
            return false;
        }

        bool[] pattern = ssvepFramePatterns[patternIndex];
        if (pattern == null || pattern.Length == 0)
        {
            return false;
        }

        patternLength = ssvepPatternLengths[patternIndex];
        if (patternLength <= 0)
        {
            return false;
        }

        int idx = frameCounter % patternLength;
        if (idx < 0) idx += patternLength;
        cursorBeforeAdvance = idx;
        isOn = pattern[idx];
        return true;
    }

    void StopCurrentStimulationVisuals(bool sendDecodeEndTrigger)
    {
        StopAllCoroutines();
        StopGlobalFrameTick();
        FlushSequenceLogBufferToCsv("stop_current_visuals");
        ResetGlobalFrameCounter();
        SetAllFramesAlpha(0f);
        SetAllImageAlpha(1f);
        if (stimulusPanel != null) stimulusPanel.SetActive(false);

        if (sendDecodeEndTrigger && visualMode == VisualMode.Decode && decodeStimStarted && currentTargetId > 0)
        {
            SendDecodeMarker(200 + Mathf.Max(1, currentTargetId));
        }

        if (visualMode == VisualMode.Pretrain)
        {
            RestoreDefaultImageTextures();
        }

        decodeStimStarted = false;
        visualMode = VisualMode.None;
    }

    void StopDecodeStimulationKeepVisuals(bool sendDecodeEndTrigger)
    {
        StopAllCoroutines();
        StopGlobalFrameTick();
        FlushSequenceLogBufferToCsv("stop_decode_keep_visuals");
        ResetGlobalFrameCounter();
        SetAllFramesAlpha(0f);
        SetAllImageAlpha(1f);
        if (stimulusPanel != null) stimulusPanel.SetActive(true);

        if (sendDecodeEndTrigger && visualMode == VisualMode.Decode && decodeStimStarted && currentTargetId > 0)
        {
            SendDecodeMarker(200 + Mathf.Max(1, currentTargetId));
        }

        decodeStimStarted = false;
        visualMode = VisualMode.Decode;
    }

    void ResetDecodeBatchState(bool clearQueues)
    {
        if (clearQueues)
        {
            pendingImageData.Clear();
            pendingImageSizes.Clear();
            pendingImageFrameIds.Clear();
        }

        foreach (var tex in receivedTextures)
        {
            if (tex != null) Destroy(tex);
        }

        receivedTextures = new List<Texture2D>(new Texture2D[6]);
        receivedImgIndices.Clear();
        isBatchCompleted = false;
        batchStartedByCommand = false;
        batchTrialId = -1;
        expectedBatchImageCount = 6;
        UpdateActiveDecodeSlotCount(6);
        currentTrialId = -1;
        currentTargetId = -1;
    }

    bool TryParseDecodeFrameMeta(string frameId, out int trialId, out int imgIdx, out int targetId)
    {
        trialId = -1;
        imgIdx = -1;
        targetId = -1;
        if (string.IsNullOrEmpty(frameId)) return false;

        string[] parts = frameId.Split(';');
        foreach (string part in parts)
        {
            string[] kv = part.Split('=');
            if (kv.Length != 2) continue;
            string key = kv[0].Trim().ToLowerInvariant();
            string val = kv[1].Trim();

            if (key == "trial") int.TryParse(val, out trialId);
            else if (key == "img") int.TryParse(val, out imgIdx);
            else if (key == "target") int.TryParse(val, out targetId);
        }

        // Decode image metadata now uses 0-based image index (img=0..5).
        return trialId > 0 && imgIdx >= 0 && imgIdx < 6 && targetId > 0;
    }

    bool TryParseCommand(
        string frameId,
        out string cmd,
        out int trialId,
        out int targetId,
        out int count
    )
    {
        cmd = "";
        trialId = -1;
        targetId = -1;
        count = -1;
        if (string.IsNullOrEmpty(frameId)) return false;

        string[] parts = frameId.Split(';');
        foreach (string part in parts)
        {
            string[] kv = part.Split('=');
            if (kv.Length != 2) continue;
            string key = kv[0].Trim().ToLowerInvariant();
            string val = kv[1].Trim();

            if (key == "cmd") cmd = val.ToLowerInvariant();
            else if (key == "trial") int.TryParse(val, NumberStyles.Integer, CultureInfo.InvariantCulture, out trialId);
            else if (key == "target") int.TryParse(val, NumberStyles.Integer, CultureInfo.InvariantCulture, out targetId);
            else if (key == "count") int.TryParse(val, NumberStyles.Integer, CultureInfo.InvariantCulture, out count);
        }

        return !string.IsNullOrEmpty(cmd);
    }

    int NormalizeDecodeBatchCount(int count)
    {
        if (count <= 0)
        {
            return 6;
        }
        return Mathf.Clamp(count, 1, 6);
    }

    void UpdateActiveDecodeSlotCount(int count)
    {
        activeDecodeSlotCount = NormalizeDecodeBatchCount(count);
        activeDecodeFrameIndices.Clear();
        for (int i = 0; i < activeDecodeSlotCount && i < decodeImageIndices.Length; i++)
        {
            activeDecodeFrameIndices.Add(decodeImageIndices[i]);
        }
        // Confirm/rollback slots (3/7) are always available for decode flashing,
        // independent of dynamic image batch size.
        activeDecodeFrameIndices.Add(3);
        activeDecodeFrameIndices.Add(7);
    }

    bool IsDecodeFrameActive(int frameIndex)
    {
        return activeDecodeFrameIndices.Contains(frameIndex);
    }

    void BeginDecodeBatch(int trialId, int targetId, int count)
    {
        bool trialSwitch = (trialId > 0 && currentTrialId > 0 && trialId != currentTrialId);
        if (decodeStimStarted || trialSwitch)
        {
            StopDecodeStimulationKeepVisuals(sendDecodeEndTrigger: true);
        }

        ResetDecodeBatchState(clearQueues: false);

        expectedBatchImageCount = NormalizeDecodeBatchCount(count);
        UpdateActiveDecodeSlotCount(expectedBatchImageCount);
        batchStartedByCommand = true;

        if (trialId > 0)
        {
            currentTrialId = trialId;
            batchTrialId = trialId;
        }

        if (targetId > 0)
        {
            currentTargetId = targetId;
        }
    }

    int TargetIdToFrameIndex(int targetId)
    {
        if (targetId <= 0 || targetId > targetIndices.Length)
        {
            return -1;
        }
        return targetIndices[targetId - 1];
    }

    void SendDecodeMarker(int value)
    {
        if (udpSender == null) return;
        try
        {
            byte[] data = new byte[] { (byte)Mathf.Clamp(value, 0, 255) };
            udpSender.Send(data, data.Length, decodeTriggerTargetIP, decodeTriggerTargetPort);
        }
        catch (Exception e)
        {
            Debug.LogWarning("Send decode marker failed: " + e.Message);
        }
    }

    void SendTrainTrialStart()
    {
        if (udpSender == null || currentTrialId <= 0 || currentTargetId <= 0) return;
        try
        {
            string payload = $"trial_start={currentTrialId};target={currentTargetId}";
            byte[] data = Encoding.UTF8.GetBytes(payload);
            udpSender.Send(data, data.Length, trainTriggerTargetIP, trainTriggerTargetPort);
        }
        catch (Exception e)
        {
            Debug.LogWarning("Send train trial_start failed: " + e.Message);
        }
    }

    void SetAllFramesAlpha(float alpha)
    {
        for (int i = 0; i < frameUIs.Length; i++)
        {
            frameUIs[i].color = new Color(1f, 1f, 1f, alpha);
        }
    }

    void SetAllImageAlpha(float alpha)
    {
        if (imageUIs == null) return;
        for (int i = 0; i < imageUIs.Length; i++)
        {
            if (imageUIs[i] == null) continue;
            Color baseColor = imageUIs[i].color;
            imageUIs[i].color = new Color(baseColor.r, baseColor.g, baseColor.b, alpha);
        }
    }

    void PrepareDecodeVisuals()
    {
        visualMode = VisualMode.Decode;
        StopAllCoroutines();
        if (stimulusPanel != null) stimulusPanel.SetActive(true);
        SetAllFramesAlpha(0f);
        SetAllImageAlpha(1f);
    }

    void CacheDefaultImageTextures()
    {
        defaultImageTextures = new Texture[imageUIs.Length];
        for (int i = 0; i < imageUIs.Length; i++)
        {
            defaultImageTextures[i] = imageUIs[i] != null ? imageUIs[i].texture : null;
        }
    }

    void RestoreDefaultImageTextures()
    {
        if (defaultImageTextures == null || imageUIs == null) return;
        int n = Mathf.Min(defaultImageTextures.Length, imageUIs.Length);
        for (int i = 0; i < n; i++)
        {
            if (imageUIs[i] == null) continue;
            imageUIs[i].texture = defaultImageTextures[i];
            imageUIs[i].gameObject.SetActive(true);
        }
    }

    void InitializePretrainSolidTextures()
    {
        redImageTexture = CreateSolidTexture(new Color(1f, 0f, 0f, 1f));
        whiteImageTexture = CreateSolidTexture(new Color(1f, 1f, 1f, 1f));
    }

    Texture2D CreateSolidTexture(Color color)
    {
        Texture2D tex = new Texture2D(2, 2, TextureFormat.RGB24, false);
        Color[] pixels = new Color[] { color, color, color, color };
        tex.SetPixels(pixels);
        tex.Apply();
        return tex;
    }

    void ApplyPretrainTargetLayout(int targetId)
    {
        int targetIndex = TargetIdToFrameIndex(targetId);
        for (int i = 0; i < imageUIs.Length; i++)
        {
            if (imageUIs[i] == null) continue;
            imageUIs[i].gameObject.SetActive(true);
            imageUIs[i].texture = (i == targetIndex) ? redImageTexture : whiteImageTexture;
        }
    }

    void OnDestroy()
    {
        StopCurrentStimulationVisuals(sendDecodeEndTrigger: false);
        FlushSequenceLogBufferToCsv("on_destroy");
        if (redImageTexture != null) Destroy(redImageTexture);
        if (whiteImageTexture != null) Destroy(whiteImageTexture);
        decodeSubscription?.Dispose();
        decodeCommandSubscription?.Dispose();
        trainSubscription?.Dispose();
        (node as IDisposable)?.Dispose();
        udpSender?.Close();
    }

    void ResetSequenceLogSession()
    {
        loggedFramesByTarget.Clear();
        RebuildSequenceLogTargetSet();
    }

    void RebuildSequenceLogTargetSet()
    {
        debugTargetIdSet.Clear();
        logAllTargets = true;

        if (string.IsNullOrWhiteSpace(debugTargetIdsToLog))
        {
            return;
        }

        string[] tokens = debugTargetIdsToLog.Split(',');
        bool hasAny = false;
        foreach (string raw in tokens)
        {
            string token = raw.Trim();
            if (token.Length == 0)
            {
                continue;
            }
            if (!int.TryParse(token, NumberStyles.Integer, CultureInfo.InvariantCulture, out int targetId))
            {
                continue;
            }
            if (targetId <= 0 || targetId > targetIndices.Length)
            {
                continue;
            }
            debugTargetIdSet.Add(targetId);
            hasAny = true;
        }

        logAllTargets = !hasAny;
    }

    bool ShouldLogTargetFrame(int targetId)
    {
        if (!enableSequenceLogging)
        {
            return false;
        }
        if (!logAllTargets && !debugTargetIdSet.Contains(targetId))
        {
            return false;
        }

        if (!loggedFramesByTarget.TryGetValue(targetId, out int count))
        {
            count = 0;
        }

        if (logFirstNFrames > 0 && count >= logFirstNFrames)
        {
            return false;
        }

        loggedFramesByTarget[targetId] = count + 1;
        return true;
    }

    void MaybeLogSequenceFrame(
        int targetId,
        int patternLength,
        int cursorBeforeAdvance,
        bool isOn
    )
    {
        if (!ShouldLogTargetFrame(targetId))
        {
            return;
        }

        sequenceLogBuffer
            .Append(Time.frameCount).Append(',')
            .Append(currentTrialId).Append(',')
            .Append(visualMode.ToString()).Append(',')
            .Append(targetId).Append(',')
            .Append(patternLength).Append(',')
            .Append(cursorBeforeAdvance).Append(',')
            .Append(isOn ? 1 : 0).Append('\n');
        sequenceLogPendingRows++;
    }

    void FlushSequenceLogBufferToCsv(string reason)
    {
        if (sequenceLogBuffer.Length == 0)
        {
            return;
        }

        try
        {
            string fileName = string.IsNullOrWhiteSpace(sequenceLogFileName)
                ? "ssvep_sequence_log.csv"
                : sequenceLogFileName.Trim();
            if (!fileName.EndsWith(".csv", StringComparison.OrdinalIgnoreCase))
            {
                fileName += ".csv";
            }

            string fullPath = Path.Combine(Application.persistentDataPath, fileName);
            bool writeHeader = !File.Exists(fullPath) || new FileInfo(fullPath).Length == 0;

            using (StreamWriter writer = new StreamWriter(fullPath, append: true, Encoding.UTF8))
            {
                if (writeHeader)
                {
                    writer.WriteLine("Time.frameCount,currentTrialId,visualMode,targetId,patternLength,cursorBeforeAdvance,isOn");
                }
                writer.Write(sequenceLogBuffer.ToString());
            }

            Debug.Log(
                $"[SSVEPSequenceLog] Flushed {sequenceLogPendingRows} rows to {fullPath}, reason={reason}"
            );
            sequenceLogBuffer.Clear();
            sequenceLogPendingRows = 0;
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[SSVEPSequenceLog] Flush failed ({reason}): {e.Message}");
        }
    }
}
