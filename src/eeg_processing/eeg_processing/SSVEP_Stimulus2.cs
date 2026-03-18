using UnityEngine;
using UnityEngine.UI;
using ROS2;
using sensor_msgs.msg;
using RosImage = sensor_msgs.msg.Image;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Net.Sockets;
using System.Text;

/// <summary>
/// Unity 侧统一 SSVEP 显示器（解码 + 预训练）。
/// - 解码模式：接收 ROS 图像批次，按 ROS 指令开始闪烁并回传 trial_started。
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
    public bool useManualRefreshRate = true;
    [Min(1)] public int manualRefreshRateHz = 165;

    // 可见目标对应频率：target_id 1..8 -> UI 索引 0..7
    public float[] ssvepFrequencies = { 8f, 10f, 12f, 15f, 20f, 30f, 40f, 45f };

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

    private Coroutine[] ssvepCoroutines;

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

        QualitySettings.vSyncCount = 0;
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

        stimulusPanel.SetActive(false);
        decodeStimStarted = false;
        SetAllFramesAlpha(0f);
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

        if (currentTrialId < 0)
        {
            currentTrialId = trialId;
            currentTargetId = targetId;
        }

        if (trialId != currentTrialId)
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

        if (imgIdx < 1 || imgIdx > 6 || receivedImgIndices.Contains(imgIdx))
        {
            return;
        }

        Texture2D tex = BgrToTexture(data, width, height);
        receivedTextures[imgIdx - 1] = tex;
        receivedImgIndices.Add(imgIdx);

        if (!isBatchCompleted && receivedImgIndices.Count >= 6)
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
            Texture2D tex = receivedTextures[i];
            if (tex != null)
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
        if (stimulusPanel != null) stimulusPanel.SetActive(true);
        SetAllFramesAlpha(0f);

        ssvepCoroutines = new Coroutine[targetIndices.Length];
        for (int i = 0; i < targetIndices.Length; i++)
        {
            int frameIdx = targetIndices[i];
            if (frameIdx < frameUIs.Length)
            {
                ssvepCoroutines[i] = StartCoroutine(SSVEPFlash(frameIdx, ssvepFrequencies[i], false));
            }
        }

        decodeStimStarted = true;
        SendDecodeMarker(100 + Mathf.Max(1, currentTargetId));
        SendDecodeTrialStarted();
    }

    // --------------------------------------
    // 解码/训练命令处理
    // --------------------------------------
    void HandleDecodeCommand(string frameId)
    {
        if (!TryParseCommand(frameId, out string cmd, out int trialId, out int targetId))
        {
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
        if (!TryParseCommand(frameId, out string cmd, out int trialId, out int targetId))
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
        if (stimulusPanel != null) stimulusPanel.SetActive(true);

        ApplyPretrainTargetLayout(currentTargetId);
        SetAllFramesAlpha(0f);
        ssvepCoroutines = new Coroutine[targetIndices.Length];
        for (int i = 0; i < targetIndices.Length; i++)
        {
            int frameIdx = targetIndices[i];
            if (frameIdx < frameUIs.Length)
            {
                bool isTarget = (i + 1) == currentTargetId;
                ssvepCoroutines[i] = StartCoroutine(SSVEPFlash(frameIdx, ssvepFrequencies[i], isTarget));
            }
        }

        SendTrainTrialStart();
    }

    IEnumerator SSVEPFlash(int frameUiIndex, float frequency, bool keepRedTarget)
    {
        while (true)
        {
            bool isOn = false;
            if (frequency > 0f)
            {
                double period = 1.0 / frequency;
                isOn = (Time.unscaledTimeAsDouble % period) < (period * 0.5);
            }

            if (keepRedTarget)
            {
                frameUIs[frameUiIndex].color = isOn
                    ? new Color(1f, 0f, 0f, 1f)
                    : new Color(1f, 0f, 0f, 0.25f);
            }
            else
            {
                frameUIs[frameUiIndex].color = isOn
                    ? new Color(1f, 1f, 1f, 1f)
                    : new Color(1f, 1f, 1f, 0.1f);
            }

            yield return null;
        }
    }

    void StopCurrentStimulationVisuals(bool sendDecodeEndTrigger)
    {
        StopAllCoroutines();
        SetAllFramesAlpha(0f);
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
        SetAllFramesAlpha(0f);
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
            else if (key == "image_id") int.TryParse(val, out targetId);
        }

        return trialId > 0 && imgIdx > 0 && targetId > 0;
    }

    bool TryParseCommand(string frameId, out string cmd, out int trialId, out int targetId)
    {
        cmd = "";
        trialId = -1;
        targetId = -1;
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
        }

        return !string.IsNullOrEmpty(cmd);
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

    void SendDecodeTrialStarted()
    {
        if (udpSender == null || currentTrialId <= 0) return;
        try
        {
            string ack = $"trial_started={currentTrialId}";
            byte[] payload = Encoding.UTF8.GetBytes(ack);
            udpSender.Send(payload, payload.Length, decodeAckTargetIP, decodeAckPort);
        }
        catch (Exception e)
        {
            Debug.LogWarning("Send decode trial_started failed: " + e.Message);
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

    void PrepareDecodeVisuals()
    {
        visualMode = VisualMode.Decode;
        StopAllCoroutines();
        if (stimulusPanel != null) stimulusPanel.SetActive(true);
        SetAllFramesAlpha(0f);
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
        if (redImageTexture != null) Destroy(redImageTexture);
        if (whiteImageTexture != null) Destroy(whiteImageTexture);
        decodeSubscription?.Dispose();
        decodeCommandSubscription?.Dispose();
        trainSubscription?.Dispose();
        (node as IDisposable)?.Dispose();
        udpSender?.Close();
    }
}
