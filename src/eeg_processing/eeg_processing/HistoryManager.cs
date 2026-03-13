using UnityEngine;
using UnityEngine.UI;
using Unity.Robotics.ROSTCPConnector;
using RosColor = RosMessageTypes.Sensor.ImageMsg; // 需确保已导入 ROS 消息定义
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

public class HistoryManager : MonoBehaviour
{
    [Header("UI 引用")]
    public Transform contentTransform;    // 拖入你的 Content 节点
    public GameObject iconPrefab;        // 拖入你做好的 History_Icon 预制体
    public ScrollRect scrollRect;        // 拖入你的 Scroll View 节点

    [Header("ROS 配置")]
    public string topicName = "/history_image";
    public bool clearInitialContentChildren = true;

    [Header("History UDP 控制通道（独立于 SSVEP 9999/10000/10001）")]
    public string historyUdpBindIP = "0.0.0.0";
    public int historyUdpBindPort = 12001;

    private readonly List<GameObject> historyIcons = new List<GameObject>();
    private readonly List<int> historyIds = new List<int>();
    private UdpClient udpReceiver;
    private Thread udpThread;
    private volatile bool udpRunning = false;

    void Start()
    {
        MainThreadDispatcher.Initialize();
        if (clearInitialContentChildren)
        {
            ClearInitialContentChildren();
        }
        StartUdpControlListener();
        // 注册订阅
        ROSConnection.GetOrCreateInstance().Subscribe<RosColor>(topicName, OnImageReceived);
    }

    void OnDestroy()
    {
        StopUdpControlListener();
    }

    void OnImageReceived(RosColor msg)
    {
        // ROS 回调可能不在 Unity 主线程，先复制数据再派发到主线程处理。
        int width = (int)msg.width;
        int height = (int)msg.height;
        string encoding = msg.encoding;
        string frameId = msg.header != null ? msg.header.frame_id : "";
        int historyId = ParseHistoryId(frameId);
        byte[] rawDataCopy = (byte[])msg.data.Clone();

        MainThreadDispatcher.Execute(() =>
        {
            if (width <= 0 || height <= 0)
            {
                Debug.LogWarning($"[HistoryManager] 无效图像尺寸: {width}x{height}");
                return;
            }

            if (rawDataCopy.Length != width * height * 3)
            {
                Debug.LogWarning(
                    $"[HistoryManager] 图像长度不匹配，encoding={encoding}, data={rawDataCopy.Length}, expected={width * height * 3}");
                return;
            }

            // 1. 将 ROS 原始数据转换为 Texture2D
            // 注意：history_sender 使用 bgr8，Unity TextureFormat.RGB24 需要 RGB 顺序。
            Texture2D tex = new Texture2D(width, height, TextureFormat.RGB24, false);

            if (encoding == "bgr8")
            {
                for (int i = 0; i < rawDataCopy.Length; i += 3)
                {
                    byte temp = rawDataCopy[i]; // B
                    rawDataCopy[i] = rawDataCopy[i + 2]; // R
                    rawDataCopy[i + 2] = temp;
                }
            }
            else if (encoding != "rgb8")
            {
                Debug.LogWarning($"[HistoryManager] 暂不支持的图像编码: {encoding}");
                return;
            }

            tex.LoadRawTextureData(rawDataCopy);
            tex.Apply();

            // 2. 实例化 UI 元素
            GameObject newIcon = Instantiate(iconPrefab, contentTransform, false);
            bool bound = TryBindTextureToIcon(newIcon, tex);
            if (!bound)
            {
                Debug.LogWarning("[HistoryManager] 未找到可显示图片的 RawImage/Image 组件，请检查 History_Item 预制体。");
            }
            historyIcons.Add(newIcon);
            historyIds.Add(historyId);

            // 3. 自动滚动到最右侧
            StartCoroutine(ScrollToRight());
        });
    }

    void StartUdpControlListener()
    {
        try
        {
            IPAddress ip = IPAddress.Parse(historyUdpBindIP);
            udpReceiver = new UdpClient(new IPEndPoint(ip, historyUdpBindPort));
            udpRunning = true;
            udpThread = new Thread(UdpReceiveLoop);
            udpThread.IsBackground = true;
            udpThread.Start();
            Debug.Log($"[HistoryManager] UDP 控制监听已启动: {historyUdpBindIP}:{historyUdpBindPort}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[HistoryManager] UDP 控制监听启动失败: {e.Message}");
        }
    }

    void StopUdpControlListener()
    {
        udpRunning = false;
        try
        {
            udpReceiver?.Close();
        }
        catch {}
        if (udpThread != null && udpThread.IsAlive)
        {
            udpThread.Join(300);
        }
    }

    void UdpReceiveLoop()
    {
        IPEndPoint remote = new IPEndPoint(IPAddress.Any, 0);
        while (udpRunning)
        {
            try
            {
                byte[] data = udpReceiver.Receive(ref remote);
                string text = Encoding.UTF8.GetString(data);
                MainThreadDispatcher.Execute(() => HandleUdpCommand(text));
            }
            catch (SocketException)
            {
                if (!udpRunning) break;
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"[HistoryManager] UDP 接收异常: {e.Message}");
            }
        }
    }

    void HandleUdpCommand(string text)
    {
        HistoryCommand command = null;
        try
        {
            command = JsonUtility.FromJson<HistoryCommand>(text);
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"[HistoryManager] UDP 控制命令解析异常: {e.Message}, raw={text}");
            return;
        }

        if (command == null || string.IsNullOrEmpty(command.cmd))
        {
            Debug.LogWarning($"[HistoryManager] 无法解析 UDP 控制命令: {text}");
            return;
        }

        string cmd = command.cmd.ToLowerInvariant();
        if (cmd == "delete_last")
        {
            DeleteLast();
            return;
        }
        if (cmd == "clear" || cmd == "delete_all")
        {
            ClearAll();
            return;
        }
        if (cmd == "delete_id")
        {
            DeleteById(command.id);
            return;
        }

        Debug.LogWarning($"[HistoryManager] 未知控制命令: {cmd}");
    }

    void DeleteLast()
    {
        if (historyIcons.Count == 0) return;

        int last = historyIcons.Count - 1;
        GameObject go = historyIcons[last];
        historyIcons.RemoveAt(last);
        historyIds.RemoveAt(last);
        if (go != null) Destroy(go);
    }

    void ClearAll()
    {
        for (int i = 0; i < historyIcons.Count; i++)
        {
            if (historyIcons[i] != null) Destroy(historyIcons[i]);
        }
        historyIcons.Clear();
        historyIds.Clear();
    }

    void ClearInitialContentChildren()
    {
        if (contentTransform == null) return;
        for (int i = contentTransform.childCount - 1; i >= 0; i--)
        {
            Transform child = contentTransform.GetChild(i);
            Destroy(child.gameObject);
        }
        historyIcons.Clear();
        historyIds.Clear();
    }

    void DeleteById(int id)
    {
        if (id <= 0) return;
        int index = historyIds.IndexOf(id);
        if (index < 0) return;

        GameObject go = historyIcons[index];
        historyIcons.RemoveAt(index);
        historyIds.RemoveAt(index);
        if (go != null) Destroy(go);
    }

    int ParseHistoryId(string frameId)
    {
        if (string.IsNullOrEmpty(frameId)) return -1;
        if (!frameId.StartsWith("hist_id=")) return -1;
        string raw = frameId.Substring("hist_id=".Length);
        int id;
        if (int.TryParse(raw, out id)) return id;
        return -1;
    }

    bool TryBindTextureToIcon(GameObject icon, Texture2D tex)
    {
        RawImage rawImage = icon.GetComponentInChildren<RawImage>(true);
        if (rawImage != null)
        {
            rawImage.texture = tex;
            rawImage.color = Color.white;
            return true;
        }

        Image image = icon.GetComponentInChildren<Image>(true);
        if (image != null)
        {
            Sprite sprite = Sprite.Create(
                tex,
                new Rect(0, 0, tex.width, tex.height),
                new Vector2(0.5f, 0.5f));
            image.sprite = sprite;
            image.color = Color.white;
            image.preserveAspect = true;
            return true;
        }

        return false;
    }

    IEnumerator ScrollToRight()
    {
        // 必须等待本帧渲染结束，因为 Layout Group 需要时间计算新的 Content 宽度
        yield return new WaitForEndOfFrame();
        Canvas.ForceUpdateCanvases();
        
        // horizontalNormalizedPosition = 1 代表最右侧
        scrollRect.horizontalNormalizedPosition = 1f;
    }
}

[System.Serializable]
public class HistoryCommand
{
    public string cmd;
    public int id;
}

// 简易主线程调度器（如果你的 ROS 插件没有自带，需要一个来处理异步消息）
public class MainThreadDispatcher : MonoBehaviour
{
    private static readonly Queue<System.Action> _queue = new Queue<System.Action>();
    private static MainThreadDispatcher _instance;
    private static readonly object _lockObj = new object();

    public static void Initialize()
    {
        if (_instance != null) return;
        GameObject go = new GameObject("MainThreadDispatcher");
        _instance = go.AddComponent<MainThreadDispatcher>();
        DontDestroyOnLoad(go);
    }

    public static void Execute(System.Action action)
    {
        if (action == null) return;
        lock (_lockObj)
        {
            _queue.Enqueue(action);
        }
    }

    void Update()
    {
        while (true)
        {
            System.Action action = null;
            lock (_lockObj)
            {
                if (_queue.Count == 0) break;
                action = _queue.Dequeue();
            }
            action?.Invoke();
        }
    }
}
