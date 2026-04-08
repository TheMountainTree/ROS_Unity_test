using System.Text;
using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std; // 假设使用标准字符串消息，如果用自定义消息请替换

public class LLMStreamManager : MonoBehaviour
{
    [Header("UI References (UI 组件绑定)")]
    [Tooltip("刚才配置好的，位于 Content 下的 Text (TMP) 组件")]
    public TextMeshProUGUI llmText;
    
    [Tooltip("最外层的 Scroll View 组件")]
    public ScrollRect scrollRect;

    [Header("ROS Settings (ROS 配置)")]
    public string llmTopic = "/llm_output_stream";

    // 使用 StringBuilder 进行高效的字符串拼接，这是流式输出性能优化的核心
    private StringBuilder textBuffer = new StringBuilder();
    
    // 标记是否需要刷新 UI
    private bool isTextDirty = false;

    void Start()
    {
        // 1. 初始化 ROS 订阅
        ROSConnection.GetOrCreateInstance().Subscribe<StringMsg>(llmTopic, OnLLMMessageReceived);
        
        // 2. 清空测试文本
        if (llmText != null) llmText.text = "等待系统接入...\n";
    }

    /// <summary>
    /// ROS 消息接收回调
    /// </summary>
    private void OnLLMMessageReceived(StringMsg msg)
    {
        // 【核心逻辑分支】: 你的 ROS 节点是如何发送数据的？
        
        // 情况 A（推荐）：如果你的 ROS 每次只发送新生成的“一个词/Token” (增量模式)
        textBuffer.Append(msg.data); 
        
        // 情况 B：如果你的 ROS 每次都会发送“从头到尾完整的整段话” (全量模式)
        // 请注释掉上面的 A，取消下面两行的注释：
        // textBuffer.Clear();
        // textBuffer.Append(msg.data);

        // 标记 UI 需要更新
        isTextDirty = true;
    }

    void Update()
    {
        // 在 Update 中统一处理 UI 刷新，避免高频回调卡死主线程
        if (isTextDirty)
        {
            // 将缓冲区的文本推送到 UI
            llmText.text = textBuffer.ToString();
            isTextDirty = false;

            // 触发平滑滚动逻辑
            StartCoroutine(UpdateScrollPosition());
        }
    }

    /// <summary>
    /// 智能滚动协程
    /// </summary>
    private IEnumerator UpdateScrollPosition()
    {
        // 【关键步骤】必须等待这一帧结束！
        // 因为给 llmText.text 赋值后，Unity 的 Content Size Fitter 需要一帧的时间来重新计算 Content 的高度
        yield return new WaitForEndOfFrame();

        if (scrollRect != null)
        {
            // 【UX 细节】判断用户当前的浏览意图：
            // 如果 verticalNormalizedPosition < 0.1f，说明用户当前视线就在最底部，我们让它自动跟随新字。
            // 如果 > 0.1f，说明用户正在往上翻看历史记录，此时我们“不要”打扰用户，让他继续看。
            if (scrollRect.verticalNormalizedPosition <= 0.15f)
            {
                // 0 代表最底部，1 代表最顶部
                scrollRect.verticalNormalizedPosition = 0f;
            }
        }
    }
}