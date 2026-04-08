using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using TMPro;
using RosMessageTypes.Std;
using Unity.Robotics.ROSTCPConnector;
using UnityEngine;
using UnityEngine.UI;

[Serializable]
public class LLMStreamEvent
{
    public string type;
    public string stage;
    public string text;
}

public class LLMStreamManager : MonoBehaviour
{
    [Header("UI References (UI 组件绑定)")]
    public TextMeshProUGUI llmText;
    public ScrollRect scrollRect;

    [Header("ROS Settings (ROS 配置)")]
    public string llmTopic = "/llm_output_stream";

    private readonly object queueLock = new object();
    private readonly Queue<string> pendingPayloads = new Queue<string>();
    private readonly StringBuilder textBuffer = new StringBuilder();
    private bool isTextDirty = false;

    void Start()
    {
        ROSConnection.GetOrCreateInstance().Subscribe<StringMsg>(llmTopic, OnLLMMessageReceived);
        if (llmText != null)
        {
            llmText.enableWordWrapping = true;
            llmText.overflowMode = TextOverflowModes.Overflow;
            llmText.text = ComposeRichText("等待系统接入...\n");
        }
    }

    private void OnLLMMessageReceived(StringMsg msg)
    {
        string payload = msg != null ? msg.data : "";
        if (string.IsNullOrEmpty(payload))
        {
            return;
        }
        lock (queueLock)
        {
            pendingPayloads.Enqueue(payload);
        }
    }

    void Update()
    {
        lock (queueLock)
        {
            while (pendingPayloads.Count > 0)
            {
                HandleIncomingPayload(pendingPayloads.Dequeue());
            }
        }

        if (isTextDirty && llmText != null)
        {
            llmText.text = ComposeRichText(textBuffer.ToString());
            isTextDirty = false;
            StartCoroutine(UpdateScrollPosition());
        }
    }

    private void HandleIncomingPayload(string payload)
    {
        if (string.IsNullOrWhiteSpace(payload))
        {
            return;
        }

        if (TryParseEvent(payload, out LLMStreamEvent evt))
        {
            ApplyStreamEvent(evt);
            return;
        }

        AppendText(payload);
    }

    private bool TryParseEvent(string payload, out LLMStreamEvent evt)
    {
        evt = null;
        string trimmed = payload.Trim();
        if (!trimmed.StartsWith("{") || !trimmed.EndsWith("}"))
        {
            return false;
        }

        try
        {
            evt = JsonUtility.FromJson<LLMStreamEvent>(trimmed);
            return evt != null && !string.IsNullOrEmpty(evt.type);
        }
        catch
        {
            evt = null;
            return false;
        }
    }

    private void ApplyStreamEvent(LLMStreamEvent evt)
    {
        string evtType = evt.type != null ? evt.type.Trim().ToLowerInvariant() : "";
        switch (evtType)
        {
            case "reset":
                textBuffer.Clear();
                if (!string.IsNullOrEmpty(evt.text))
                {
                    textBuffer.Append(evt.text);
                }
                isTextDirty = true;
                break;
            case "append":
                AppendText(evt.text);
                break;
            case "error":
                if (!string.IsNullOrEmpty(evt.text))
                {
                    AppendText("\n[LLM] ");
                    AppendText(evt.text);
                }
                break;
            case "done":
                if (!string.IsNullOrEmpty(evt.text))
                {
                    AppendText(evt.text);
                }
                break;
            default:
                if (!string.IsNullOrEmpty(evt.text))
                {
                    AppendText(evt.text);
                }
                break;
        }
    }

    private void AppendText(string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return;
        }
        textBuffer.Append(text);
        isTextDirty = true;
    }

    private string ComposeRichText(string body)
    {
        string safeBody = EscapeRichText(body);
        return $"<b>LLM Output</b>\n<color=#E8ECF1>{safeBody}</color>";
    }

    private string EscapeRichText(string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return string.Empty;
        }
        return text
            .Replace("&", "&amp;")
            .Replace("<", "&lt;")
            .Replace(">", "&gt;");
    }

    private IEnumerator UpdateScrollPosition()
    {
        yield return new WaitForEndOfFrame();
        if (scrollRect != null)
        {
            Canvas.ForceUpdateCanvases();
            scrollRect.verticalNormalizedPosition = 0f;
        }
    }
}
