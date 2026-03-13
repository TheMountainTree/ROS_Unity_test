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

public class ROS2SSVEPTrainStimulator : MonoBehaviour
{
    public ROS2UnityComponent ros2UnityComponent;
    public RawImage[] imageUIs;  // expected length >= 8, active: 0,1,2,4,5,6
    public UnityEngine.UI.Image[] frameUIs;
    public GameObject stimulusPanel;

    public string commandTopic = "/ssvep_train_cmd";
    public bool useReliableQos = true;

    public float[] ssvepFrequencies = { 8f, 10f, 12f, 15f, 20f, 30f };

    public string udpTriggerTargetIP = "127.0.0.1";
    public int udpTriggerTargetPort = 10001;

    private readonly int[] targetIndices = { 0, 1, 2, 4, 5, 6 };

    private UdpClient udpSender;
    private ROS2Node node;
    private ISubscription<RosImage> subscription;

    private Queue<string> pendingFrameIds = new Queue<string>();
    private object queueLock = new object();
    private Coroutine[] ssvepCoroutines;

    private int currentTrialId = -1;
    private int currentTargetId = -1;

    void Start()
    {
        if (stimulusPanel == null || imageUIs == null || frameUIs == null)
        {
            Debug.LogError("Assign imageUIs/frameUIs/stimulusPanel in Inspector.");
            enabled = false;
            return;
        }

        if (targetIndices.Length != ssvepFrequencies.Length)
        {
            Debug.LogError("ssvepFrequencies length must match visible target count (6).");
            enabled = false;
            return;
        }

        stimulusPanel.SetActive(false);
        SetAllFramesAlpha(0f);

        try
        {
            udpSender = new UdpClient();
            Debug.Log($"UDP sender initialized {udpTriggerTargetIP}:{udpTriggerTargetPort}");
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

        node = ros2UnityComponent.CreateNode("ssvep_train_stimulator");
        var qos = new QualityOfServiceProfile();
        qos.SetHistory(HistoryPolicy.QOS_POLICY_HISTORY_KEEP_LAST, 10);
        qos.SetReliability(
            useReliableQos
                ? ReliabilityPolicy.QOS_POLICY_RELIABILITY_RELIABLE
                : ReliabilityPolicy.QOS_POLICY_RELIABILITY_BEST_EFFORT
        );

        subscription = node.CreateSubscription<RosImage>(commandTopic, OnCommandReceived, qos);
        Debug.Log($"Subscribed SSVEP train command topic: {commandTopic}");
    }

    void OnCommandReceived(RosImage msg)
    {
        string frameId = msg.Header != null ? msg.Header.Frame_id : "";
        if (string.IsNullOrWhiteSpace(frameId))
        {
            return;
        }

        lock (queueLock)
        {
            pendingFrameIds.Enqueue(frameId);
        }
    }

    void Update()
    {
        lock (queueLock)
        {
            while (pendingFrameIds.Count > 0)
            {
                string frameId = pendingFrameIds.Dequeue();
                HandleCommand(frameId);
            }
        }
    }

    void HandleCommand(string frameId)
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
            StartStimulation();
            return;
        }

        if (cmd == "rest" || cmd == "stop" || cmd == "done")
        {
            StopVisuals();
            return;
        }
    }

    void ShowCue(int targetId)
    {
        StopAllCoroutines();
        if (stimulusPanel != null) stimulusPanel.SetActive(true);

        SetAllFramesAlpha(0.15f);
        int frameIdx = TargetIdToFrameIndex(targetId);
        if (frameIdx >= 0 && frameIdx < frameUIs.Length)
        {
            frameUIs[frameIdx].color = new Color(1f, 0f, 0f, 1f);
        }

        for (int i = 0; i < imageUIs.Length; i++)
        {
            if (i == 3 || i == 7)
            {
                imageUIs[i].gameObject.SetActive(false);
            }
            else
            {
                imageUIs[i].gameObject.SetActive(true);
            }
        }
    }

    void StartStimulation()
    {
        StopAllCoroutines();
        if (stimulusPanel != null) stimulusPanel.SetActive(true);

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

        SendTrialStartTrigger();
    }

    IEnumerator SSVEPFlash(int frameUiIndex, float frequency, bool isTarget)
    {
        int refresh = Screen.currentResolution.refreshRate > 0 ? Screen.currentResolution.refreshRate : 60;
        int frames = Mathf.Max(2, Mathf.RoundToInt(refresh / Mathf.Max(0.5f, frequency)));
        int halfFrames = Mathf.Max(1, frames / 2);
        int counter = 0;

        while (true)
        {
            if (isTarget)
            {
                frameUIs[frameUiIndex].color = counter < halfFrames
                    ? new Color(1f, 0f, 0f, 1f)
                    : new Color(1f, 0f, 0f, 0.25f);
            }
            else
            {
                frameUIs[frameUiIndex].color = counter < halfFrames
                    ? new Color(1f, 1f, 1f, 1f)
                    : new Color(1f, 1f, 1f, 0.1f);
            }
            counter++;
            if (counter >= frames) counter = 0;
            yield return null;
        }
    }

    void StopVisuals()
    {
        StopAllCoroutines();
        SetAllFramesAlpha(0f);
        if (stimulusPanel != null) stimulusPanel.SetActive(false);
    }

    bool TryParseCommand(string frameId, out string cmd, out int trialId, out int targetId)
    {
        cmd = "";
        trialId = -1;
        targetId = -1;

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

    void SendTrialStartTrigger()
    {
        if (udpSender == null || currentTrialId <= 0 || currentTargetId <= 0) return;

        try
        {
            string payload = $"trial_start={currentTrialId};target={currentTargetId}";
            byte[] data = Encoding.UTF8.GetBytes(payload);
            udpSender.Send(data, data.Length, udpTriggerTargetIP, udpTriggerTargetPort);
            Debug.Log($"Sent trial start trigger: {payload}");
        }
        catch (Exception e)
        {
            Debug.LogWarning("Send trial start trigger failed: " + e.Message);
        }
    }

    void SetAllFramesAlpha(float alpha)
    {
        for (int i = 0; i < frameUIs.Length; i++)
        {
            frameUIs[i].color = new Color(1f, 1f, 1f, alpha);
        }
    }

    void OnDestroy()
    {
        StopVisuals();
        subscription?.Dispose();
        (node as IDisposable)?.Dispose();
        udpSender?.Close();
    }
}
