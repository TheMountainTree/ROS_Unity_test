using UnityEngine;
using UnityEngine.UI;
using ROS2;
using sensor_msgs.msg;
using RosImage = sensor_msgs.msg.Image;
using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System;

public class ROS2SSVEPImageStimulator : MonoBehaviour
{
    public ROS2UnityComponent ros2UnityComponent;
    public RawImage[] imageUIs;  // expected length >= 8, active: 0,1,2,4,5,6
    public UnityEngine.UI.Image[] frameUIs;
    public GameObject stimulusPanel;

    public float[] ssvepFrequencies = { 8f, 10f, 12f, 15f, 20f, 30f, 40f, 45f };

    // trigger UDP (for data logger/decoder)
    public string udpTargetIP = "127.0.0.1";
    public int udpTargetPort = 9999;
    // trial-start UDP (for central controller timing start)
    public string udpAckTargetIP = "127.0.0.1";
    public int udpAckPort = 10000;
    public bool useReliableQos = true;

    private readonly int[] targetIndices = { 0, 1, 2, 4, 5, 6 };
    private readonly int[][] rows = new int[][]
    {
        new int[] {0, 1, 2, 3},
        new int[] {4, 5, 6, 7}
    };

    private readonly int[][] cols = new int[][]
    {
        new int[] {0, 4},
        new int[] {1, 5},
        new int[] {2, 6},
        new int[] {3, 7}
    };

    private UdpClient udpSender;
    private ROS2Node node;
    private ISubscription<RosImage> subscription;

    private List<Texture2D> receivedTextures = new List<Texture2D>(new Texture2D[6]);
    private bool isBatchCompleted = false;
    private int currentTrialId = -1;
    private int currentTargetId = -1;
    private HashSet<int> receivedImgIndices = new HashSet<int>(); // 1..6

    private Queue<byte[]> pendingImageData = new Queue<byte[]>();
    private Queue<(int width, int height)> pendingImageSizes = new Queue<(int, int)>();
    private Queue<string> pendingFrameIds = new Queue<string>();
    private object queueLock = new object();

    private Coroutine[] ssvepCoroutines;

    void Start()
    {
        if (stimulusPanel == null || imageUIs == null || frameUIs == null)
        {
            Debug.LogError("Assign imageUIs/frameUIs/stimulusPanel in Inspector.");
            enabled = false;
            return;
        }

        stimulusPanel.SetActive(false);
        SetAllFramesAlpha(0f);
        StartNewBatch(true);

        try
        {
            udpSender = new UdpClient();
            Debug.Log($"UDP sender initialized trigger={udpTargetIP}:{udpTargetPort}, ack={udpAckTargetIP}:{udpAckPort}");
        }
        catch (Exception e)
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
            Debug.LogError("ROS2UnityComponent not found.");
            enabled = false;
            return;
        }

        node = ros2UnityComponent.CreateNode("ssvep_image_receiver");
        var imageQos = new QualityOfServiceProfile();
        imageQos.SetHistory(HistoryPolicy.QOS_POLICY_HISTORY_KEEP_LAST, 10);
        imageQos.SetReliability(
            useReliableQos
                ? ReliabilityPolicy.QOS_POLICY_RELIABILITY_RELIABLE
                : ReliabilityPolicy.QOS_POLICY_RELIABILITY_BEST_EFFORT
        );
        Debug.Log($"SSVEP image subscriber QoS reliability: {(useReliableQos ? "RELIABLE" : "BEST_EFFORT")}");
        subscription = node.CreateSubscription<RosImage>("/image_seg", OnImageReceived, imageQos);

    }

    void OnImageReceived(RosImage msg)
    {
        if (msg.Encoding != "bgr8")
        {
            Debug.LogWarning("Expected bgr8");
            return;
        }

        byte[] dataCopy = new byte[msg.Data.Length];
        System.Buffer.BlockCopy(msg.Data, 0, dataCopy, 0, msg.Data.Length);
        lock (queueLock)
        {
            pendingImageData.Enqueue(dataCopy);
            pendingImageSizes.Enqueue(((int)msg.Width, (int)msg.Height));
            pendingFrameIds.Enqueue(msg.Header != null ? msg.Header.Frame_id : "");
        }
    }

    void Update()
    {
        lock (queueLock)
        {
            while (pendingImageData.Count > 0)
            {
                byte[] data = pendingImageData.Dequeue();
                var (w, h) = pendingImageSizes.Dequeue();
                string frameId = pendingFrameIds.Dequeue();

                int trialId, imgIdx, targetId;
                string cmd;
                if (!TryParseFrameMeta(frameId, out trialId, out imgIdx, out targetId, out cmd))
                {
                    continue;
                }

                if (cmd == "stop")
                {
                    if (currentTrialId > 0 && trialId == currentTrialId)
                    {
                        StopCurrentStimulationVisuals(sendEndTrigger: true);
                        ResetBatchState_NoLock(clearPending: false);
                    }
                    continue;
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
                        StopCurrentStimulationVisuals(sendEndTrigger: true);
                        // Already inside queueLock: reset without re-locking to avoid deadlock.
                        // Keep pending queue, otherwise we may drop the rest of this new trial's images.
                        ResetBatchState_NoLock(clearPending: false);
                        currentTrialId = trialId;
                        currentTargetId = targetId;
                    }
                    else
                    {
                        continue;
                    }
                }

                if (imgIdx < 1 || imgIdx > 6 || receivedImgIndices.Contains(imgIdx))
                {
                    continue;
                }

                Texture2D tex = new Texture2D(w, h, TextureFormat.RGB24, false);
                int pixelCount = w * h;
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

                receivedTextures[imgIdx - 1] = tex;
                receivedImgIndices.Add(imgIdx);

                if (!isBatchCompleted && receivedImgIndices.Count >= 6)
                {
                    isBatchCompleted = true;
                    AssignTexturesToUI();
                    StartStimulation();
                }
            }
        }
    }

    void AssignTexturesToUI()
    {
        if (imageUIs == null || imageUIs.Length <= 6)
        {
            Debug.LogError("imageUIs array too short");
            return;
        }

        for (int i = 0; i < targetIndices.Length; i++)
        {
            int uiIndex = targetIndices[i];
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

        if (imageUIs.Length > 3) imageUIs[3].gameObject.SetActive(false);
        if (imageUIs.Length > 7) imageUIs[7].gameObject.SetActive(false);
    }

    void StartStimulation()
    {
        if (stimulusPanel != null) stimulusPanel.SetActive(true);
        StopAllCoroutines();
        ssvepCoroutines = new Coroutine[frameUIs.Length];
        SendTrigger(100 + Mathf.Max(1, currentTargetId)); // trial start marker with target
        SendTrialStarted();

        for (int i = 0; i < frameUIs.Length; i++)
        {
            if (i < ssvepFrequencies.Length)
            {
                ssvepCoroutines[i] = StartCoroutine(SSVEPFlash(i, ssvepFrequencies[i]));
            }
        }
    }

    IEnumerator SSVEPFlash(int index, float frequency)
    {
        int frames = Mathf.Max(2, Mathf.RoundToInt(Screen.currentResolution.refreshRate / frequency));
        int halfFrames = Mathf.Max(1, frames / 2);
        int counter = 0;

        while (true)
        {
            frameUIs[index].color = counter < halfFrames
                ? new Color(1, 1, 1, 1f)
                : new Color(1, 1, 1, 0.1f);

            counter++;
            if (counter >= frames) counter = 0;
            yield return null;
        }
    }

    void StopCurrentStimulationVisuals(bool sendEndTrigger)
    {
        StopAllCoroutines();
        for (int i = 0; i < frameUIs.Length; i++)
        {
            frameUIs[i].color = new Color(1, 1, 1, 0f);
        }
        if (stimulusPanel != null) stimulusPanel.SetActive(false);
        if (sendEndTrigger && currentTargetId > 0)
        {
            SendTrigger(200 + Mathf.Max(1, currentTargetId)); // trial end marker with target
        }
    }

    void StartNewBatch(bool clearPending)
    {
        lock (queueLock)
        {
            ResetBatchState_NoLock(clearPending);
        }
    }

    void ResetBatchState_NoLock(bool clearPending)
    {
        if (clearPending)
        {
            pendingImageData.Clear();
            pendingImageSizes.Clear();
            pendingFrameIds.Clear();
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
        SetAllFramesAlpha(0f);
    }

    bool TryParseFrameMeta(string frameId, out int trialId, out int imgIdx, out int targetId, out string cmd)
    {
        trialId = -1;
        imgIdx = -1;
        targetId = -1;
        cmd = "";
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
            else if (key == "cmd") cmd = val.ToLowerInvariant();
        }

        if (cmd == "stop")
        {
            return trialId > 0;
        }
        return trialId > 0 && imgIdx > 0 && targetId > 0;
    }

    void SendTrigger(int value)
    {
        if (udpSender == null) return;
        try
        {
            byte[] data = new byte[] { (byte)Mathf.Clamp(value, 0, 255) };
            udpSender.Send(data, data.Length, udpTargetIP, udpTargetPort);
        }
        catch (Exception e)
        {
            Debug.LogWarning("Send trigger failed: " + e.Message);
        }
    }

    void SendTrialStarted()
    {
        if (udpSender == null || currentTrialId <= 0) return;
        try
        {
            string ack = $"trial_started={currentTrialId}";
            byte[] payload = System.Text.Encoding.UTF8.GetBytes(ack);
            udpSender.Send(payload, payload.Length, udpAckTargetIP, udpAckPort);
            Debug.Log($"Sent ack: {ack} to {udpAckTargetIP}:{udpAckPort}");
        }
        catch (Exception e)
        {
            Debug.LogWarning("Send trial_started failed: " + e.Message);
        }
    }

    void SetAllFramesAlpha(float alpha)
    {
        for (int i = 0; i < frameUIs.Length; i++)
        {
            frameUIs[i].color = new Color(1, 1, 1, alpha);
        }
    }

    void OnDestroy()
    {
        StopCurrentStimulationVisuals(sendEndTrigger: false);
        subscription?.Dispose();
        (node as IDisposable)?.Dispose();
        udpSender?.Close();
    }
}
