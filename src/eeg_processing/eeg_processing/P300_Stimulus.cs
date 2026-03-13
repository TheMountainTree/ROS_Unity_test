using UnityEngine;
using UnityEngine.UI;
using ROS2;
using sensor_msgs.msg;
using RosImage = sensor_msgs.msg.Image;
using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Net;
using System;

// 注意，一共4列2行，第四列的两行不是可以接受的图片，而是固定的勾和叉，按照编号，实际可以用的是0,1,2,4,5,6
public class ROS2P300ImageStimulator : MonoBehaviour
{
    public ROS2UnityComponent ros2UnityComponent; // 建议在Inspector中显式拖拽 ROS2Manager 上的组件
    public RawImage[] imageUIs;      // 实际使用索引: 0,1,2,4,5,6 (数组长度应 >= 7)
    public UnityEngine.UI.Image[] frameUIs;
    public GameObject stimulusPanel; // 拖拽 StimulusPanel 到这里

    public int stimRounds = 3;
    public float flashDuration = 0.1f;
    public float interFlashInterval = 0.075f;
    public bool useReliableQos = true;

    // === UDP Trigger Settings (替代 COM) ===
    public string udpTargetIP = "192.168.10.3"; // 接收端的 IP 地址
    public int udpTargetPort = 9999;             // 接收端监听的 UDP 端口
    public string udpAckTargetIP = "127.0.0.1"; // CentralControllerNode 所在机器IP
    public int udpAckPort = 10000;               // trial_done 握手端口

    // 要显示的6个索引（跳过3）
    private readonly int[] targetIndices = { 0, 1, 2, 4, 5, 6 };


    // 内部状态
    private List<Texture2D> receivedTextures = new List<Texture2D>();
    private bool isBatchCompleted = false;
    private int currentTrialId = -1;
    private HashSet<int> receivedImgIndices = new HashSet<int>(); // 1..6
    private bool sawMetadata = false;
    private UdpClient udpSender;
    private ROS2Node node;
    private ISubscription<RosImage> subscription;
    private bool hasReceivedAnyImage = false;
    private float startupTime = 0f;
    private bool startupWarned = false;
    public float startupNoImageWarnSeconds = 8.0f;

    private readonly int[][] rows = new int[][] {
        new int[] {0, 1, 2, 3},
        new int[] {4, 5, 6, 7}
    };

    private readonly int[][] cols = new int[][] {
        new int[] {0, 4},
        new int[] {1, 5},
        new int[] {2, 6},
        new int[] {3, 7}
    };

    void Start()
    {
         if (stimulusPanel == null || imageUIs == null || frameUIs == null)
        {
            Debug.LogError("请在 Inspector 中为 imageUIs、frameUIs 和 stimulusPanel 赋值！");
            enabled = false;
            return;
        }

        // 确保初始隐藏
        stimulusPanel.SetActive(false);
        startupTime = Time.time;

        // 初始化 frame 为透明
        SetAllFramesAlpha(0f);
        StartNewBatch(true);
        
                      
        // 初始化 UDP 发送器
        try
        {
            udpSender = new UdpClient();
            Debug.Log($"UDP sender initialized to {udpTargetIP}:{udpTargetPort}");
        }
        catch (System.Exception e)
        {
            Debug.LogError("Failed to initialize UDP client: " + e.Message);
            udpSender = null;
        }

        if (ros2UnityComponent == null)
        {
            ros2UnityComponent = FindObjectOfType<ROS2UnityComponent>();
        }
        if (ros2UnityComponent == null)
        {
            Debug.LogError("ROS2UnityComponent not found. Assign it from ROS2Manager in Inspector.");
            enabled = false;
            return;
        }

        try
        {
            node = ros2UnityComponent.CreateNode("p300_image_receiver");
        }
        catch (System.Exception e)
        {
            Debug.LogError("Failed to create ROS2 node: " + e.Message);
            enabled = false;
            return;
        }
        if (node == null)
        {
            Debug.LogError("CreateNode returned null.");
            enabled = false;
            return;
        }

        var imageQos = new QualityOfServiceProfile();
        imageQos.SetHistory(HistoryPolicy.QOS_POLICY_HISTORY_KEEP_LAST, 10);
        imageQos.SetReliability(
            useReliableQos
                ? ReliabilityPolicy.QOS_POLICY_RELIABILITY_RELIABLE
                : ReliabilityPolicy.QOS_POLICY_RELIABILITY_BEST_EFFORT
        );
        Debug.Log(
            $"P300 image subscriber QoS reliability: {(useReliableQos ? "RELIABLE" : "BEST_EFFORT")}"
        );
        try
        {
            subscription = node.CreateSubscription<RosImage>(
                "/image_seg",
                OnImageReceived,
                imageQos
            );
        }
        catch (System.Exception e)
        {
            Debug.LogError("Failed to create /image_seg subscription: " + e.Message);
            enabled = false;
            return;
        }
    }


    private Queue<byte[]> pendingImageData = new Queue<byte[]>();
    private Queue<(int width, int height)> pendingImageSizes = new Queue<(int, int)>();
    private Queue<string> pendingFrameIds = new Queue<string>();
    private object queueLock = new object(); // 用于线程安全


    void OnImageReceived(RosImage msg)
    {
        if (msg.Encoding != "bgr8")
        {
            Debug.LogWarning("Expected BGR8 encoding");
            return;
        }

        // 只做轻量级操作：复制数据到临时缓冲区
        byte[] dataCopy = new byte[msg.Data.Length];
        System.Buffer.BlockCopy(msg.Data, 0, dataCopy, 0, msg.Data.Length);

        // 线程安全地加入队列
        lock (queueLock)
        {
            pendingImageData.Enqueue(dataCopy);
            pendingImageSizes.Enqueue(((int)msg.Width, (int)msg.Height));
            pendingFrameIds.Enqueue(msg.Header != null ? msg.Header.Frame_id : "");
            hasReceivedAnyImage = true;
        }
    }


    void AssignTexturesToUI()
    {
        // 确保 imageUIs 足够长
        if (imageUIs == null || imageUIs.Length <= 6)
        {
            Debug.LogError("imageUIs array too short!");
            return;
        }

        for (int i = 0; i < targetIndices.Length; i++)
        {
            int uiIndex = targetIndices[i];
            Texture2D tex = (i < receivedTextures.Count) ? receivedTextures[i] : null;
            if (tex != null)
            {
                imageUIs[uiIndex].texture = tex;
                imageUIs[uiIndex].gameObject.SetActive(true); // 可选：激活
            }
            else
            {
                imageUIs[uiIndex].texture = null;
                imageUIs[uiIndex].gameObject.SetActive(false);
            }
        }

        // 固定占位（勾/叉）对应格子不用于图片显示，避免被白块覆盖
        if (imageUIs.Length > 3)
            imageUIs[3].gameObject.SetActive(false);
        if (imageUIs.Length > 7)
            imageUIs[7].gameObject.SetActive(false);
    }

    void StartStimulation()
    {
        if (stimulusPanel != null)
            stimulusPanel.SetActive(true);

        StopAllCoroutines();
        StartCoroutine(P300RowColumnFlashing());
    }


    public void StartNewBatch(bool clearPending = false)
    {
        if (clearPending)
        {
            lock (queueLock)
            {
                pendingImageData.Clear();
                pendingImageSizes.Clear();
                pendingFrameIds.Clear();
            }
        }

        foreach (var tex in receivedTextures)
        {
            if (tex != null) Destroy(tex);
        }
        receivedTextures = new List<Texture2D>(new Texture2D[6]);
        isBatchCompleted = false;
        currentTrialId = -1;
        receivedImgIndices.Clear();
        sawMetadata = false;

        // 清空 UI
        if (imageUIs != null)
        {
            foreach (var img in imageUIs)
            {
                if (img != null)
                {
                    img.texture = null;
                    img.gameObject.SetActive(false);
                }
            }
        }

        // 动态6个格子在下一批收到时再开启
        for (int i = 0; i < targetIndices.Length; i++)
        {
            int uiIndex = targetIndices[i];
            if (uiIndex < imageUIs.Length && imageUIs[uiIndex] != null)
            {
                imageUIs[uiIndex].gameObject.SetActive(true);
            }
        }

        if (stimulusPanel != null)
            stimulusPanel.SetActive(false);
    }

    void Update()
    {
        if (!hasReceivedAnyImage && !startupWarned && Time.time - startupTime > startupNoImageWarnSeconds)
        {
            startupWarned = true;
            Debug.LogWarning(
                $"No /image_seg received in first {startupNoImageWarnSeconds:F1}s. " +
                "Check ROS2 topic, ROS_DOMAIN_ID, and scene script references."
            );
        }

        if (receivedTextures.Count != 6)
        {
            receivedTextures = new List<Texture2D>(new Texture2D[6]);
        }

        // 消费所有待处理图像（通常一次只有一张）
        lock (queueLock)
        {
            while (pendingImageData.Count > 0 && !isBatchCompleted)
            {
                byte[] data = pendingImageData.Dequeue();
                var (w, h) = pendingImageSizes.Dequeue();
                string frameId = pendingFrameIds.Dequeue();

                int trialId = -1;
                int imgIdx = -1;
                bool hasMeta = TryParseFrameMeta(frameId, out trialId, out imgIdx);
                if (hasMeta) sawMetadata = true;
                if (hasMeta)
                {
                    if (currentTrialId < 0)
                    {
                        currentTrialId = trialId;
                    }

                    if (trialId != currentTrialId)
                    {
                        if (trialId > currentTrialId)
                        {
                            // 收到更新trial，说明旧trial未凑齐，丢弃旧trial并切到新trial
                            foreach (var oldTex in receivedTextures)
                            {
                                if (oldTex != null) Destroy(oldTex);
                            }
                            receivedTextures = new List<Texture2D>(new Texture2D[6]);
                            receivedImgIndices.Clear();
                            currentTrialId = trialId;
                        }
                        else
                        {
                            // 过期包，忽略
                            continue;
                        }
                    }

                    if (imgIdx < 1 || imgIdx > 6)
                    {
                        continue;
                    }
                    if (receivedImgIndices.Contains(imgIdx))
                    {
                        // 同一trial重复包，忽略
                        continue;
                    }
                }
                else if (sawMetadata)
                {
                    // Once metadata stream is established, ignore packets without metadata
                    // to avoid mixing frames across trials.
                    continue;
                }

                // 主线程安全：创建 Texture2D
                Texture2D tex = new Texture2D(w, h, TextureFormat.RGB24, false);

                // 转 BGR -> RGB
                int pixelCount = w * h;
                byte[] rgb = new byte[data.Length];
                for (int i = 0; i < pixelCount; i++)
                {
                    int idx = i * 3;
                    rgb[idx + 0] = data[idx + 2]; // R
                    rgb[idx + 1] = data[idx + 1]; // G
                    rgb[idx + 2] = data[idx + 0]; // B
                }

                tex.LoadRawTextureData(rgb);
                tex.Apply();

                // 存入已接收列表
                if (hasMeta)
                {
                    receivedTextures[imgIdx - 1] = tex;
                    receivedImgIndices.Add(imgIdx);
                }
                else
                {
                    // 向后兼容：没有meta时按顺序填空位
                    for (int i = 0; i < 6; i++)
                    {
                        if (receivedTextures[i] == null)
                        {
                            receivedTextures[i] = tex;
                            receivedImgIndices.Add(i + 1);
                            break;
                        }
                    }
                }

                // 如果收满6张，分配并启动刺激
                // Only start when 6 unique image slots are filled.
                if (receivedImgIndices.Count >= 6)
                {
                    Debug.Log(
                        $"Start stimulation: unique={receivedImgIndices.Count}, trial={currentTrialId}"
                    );
                    isBatchCompleted = true;
                    AssignTexturesToUI();
                    StartStimulation();
                }
            }
        }
    }

    bool TryParseFrameMeta(string frameId, out int trialId, out int imgIdx)
    {
        trialId = -1;
        imgIdx = -1;
        if (string.IsNullOrEmpty(frameId)) return false;

        string[] parts = frameId.Split(';');
        foreach (string part in parts)
        {
            string[] kv = part.Split('=');
            if (kv.Length != 2) continue;
            string k = kv[0].Trim().ToLowerInvariant();
            string v = kv[1].Trim();
            if (k == "trial")
            {
                int.TryParse(v, out trialId);
            }
            else if (k == "img")
            {
                int.TryParse(v, out imgIdx);
            }
        }
        return trialId > 0 && imgIdx > 0;
    }

    IEnumerator P300RowColumnFlashing()
    {
        List<int[]> sequence = new List<int[]>();
        for (int r = 0; r < rows.Length; r++) sequence.Add(rows[r]);
        for (int c = 0; c < cols.Length; c++) sequence.Add(cols[c]);

        for (int round = 0; round < stimRounds; round++)
        {
            List<int[]> shuffledSeq = new List<int[]>(sequence);
            ShuffleList(shuffledSeq);
            
            foreach (var group in shuffledSeq)
            {
                int triggerId = GetTriggerId(group);
                Debug.Log($"Flashing group: [{string.Join(", ", group)}]");
                // === 通过 UDP 发送 triggerId ===
                if (udpSender != null)
                {
                    try
                    {
                        byte[] data = new byte[] { (byte)triggerId }; // 只发一个字节
                        udpSender.Send(data, data.Length, udpTargetIP, udpTargetPort);
                        // Debug.Log($"Sent UDP trigger: {triggerId} to {udpTargetIP}:{udpTargetPort}");
                    }
                    catch (System.Exception e)
                    {
                        Debug.LogError("UDP send failed: " + e.Message);
                    }
                }

                SetAllFramesAlpha(0.1f);
                foreach (int idx in group)
                {
                    if (idx < frameUIs.Length)
                        frameUIs[idx].color = new Color(1, 1, 1, 1);
                }

                yield return new WaitForSeconds(flashDuration);

                SetAllFramesAlpha(0.1f);

                yield return new WaitForSeconds(interFlashInterval);
            }
        }

        SetAllFramesAlpha(0f);
        SendTrialDoneAck();
        // Start clean for next trial to avoid stale-frame replay.
        StartNewBatch(true);
    }

    void SendTrialDoneAck()
    {
        if (udpSender == null || currentTrialId <= 0) return;
        try
        {
            string ack = $"trial_done={currentTrialId}";
            byte[] payload = System.Text.Encoding.UTF8.GetBytes(ack);
            udpSender.Send(payload, payload.Length, udpAckTargetIP, udpAckPort);
            Debug.Log($"Sent ack: {ack} to {udpAckTargetIP}:{udpAckPort}");
        }
        catch (System.Exception e)
        {
            Debug.LogWarning("Failed to send trial_done ack: " + e.Message);
        }
    }

    void SetAllFramesAlpha(float alpha)
    {
        for (int i = 0; i < frameUIs.Length; i++)
        {
            frameUIs[i].color = new Color(1, 1, 1, alpha);
        }
    }

    int GetTriggerId(int[] group)
    {
        for (int i = 0; i < rows.Length; i++)
        {
            if (ArraysEqual(group, rows[i])) return i + 1;
        }
        for (int i = 0; i < cols.Length; i++)
        {
            if (ArraysEqual(group, cols[i])) return rows.Length + i + 1;
        }
        return 0;
    }

    bool ArraysEqual(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
            if (a[i] != b[i]) return false;
        return true;
    }

    void ShuffleList<T>(List<T> list)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = UnityEngine.Random.Range(0, i + 1);
            T temp = list[i];
            list[i] = list[j];
            list[j] = temp;
        }
    }

    void OnDestroy()
    {
        subscription?.Dispose();
        (node as System.IDisposable)?.Dispose();
        udpSender?.Close(); // 关闭 UDP
    }
}
