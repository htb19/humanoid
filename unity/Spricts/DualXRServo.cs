using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.Serialization;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;
using RosMessageTypes.Std;
using RosMessageTypes.BuiltinInterfaces;

public class DualXRServo_ReferenceFrameIncremental : MonoBehaviour
{
    private sealed class HandState
    {
        public Vector3 lastSampleRefPos;
        public Quaternion lastSampleRefRot = Quaternion.identity;
        public float lastSampleTime;
        public bool hasSample;
        public bool sampledActive;
        public bool wasActiveLastFrame;
        public Vector3 targetLinearRef;
        public Vector3 targetAngularRef;
        public Vector3 currentLinearRef;
        public Vector3 currentAngularRef;
    }

    [Header("ROS")]
    public ROSConnection ros;
    public string leftTopic = "/unity/left_twist_raw";
    public string rightTopic = "/unity/right_twist_raw";

    [Tooltip("这里填 Servo 命令实际表达的坐标系，比如 base_link。")]
    public string commandFrame = "base_link";

    [Header("Controller Transforms")]
    public Transform leftController;
    public Transform rightController;

    [Header("Reference Frame (recommended: XR Origin / tracking space root)")]
    [Tooltip("用于消除头部/相机整体位移影响的参考坐标系。通常填 XR Origin 节点。")]
    public Transform referenceFrame;

    [Header("Global Enable")]
    public bool enableLeft = true;
    public bool enableRight = true;

    [Header("Trigger-to-Enable")]
    [Tooltip("按住左手扳机时才发送左手速度")]
    public InputActionProperty leftTriggerAction;

    [Tooltip("按住右手扳机时才发送右手速度")]
    public InputActionProperty rightTriggerAction;

    [Tooltip("Use Quest primary buttons as the default enable buttons: X on left hand and A on right hand.")]
    public bool usePrimaryButtonsForEnable = true;

    [Range(0f, 1f)]
    [Tooltip("扳机值大于该阈值时视为使能")]
    public float triggerThreshold = 0.5f;

    [Tooltip("是否在未按扳机时持续发送零速度")]
    public bool sendZeroWhenReleased = false;

    [Header("Linear Intent")]
    public float linearScale = 0.6f;
    [FormerlySerializedAs("maxLinear")]
    public float maxLinearIntent = 0.6f;
    public float linearDeadZone = 0.008f;
    public float maxLinearAccel = 2.5f;

    [Header("Angular Intent")]
    public float angularScale = 0.8f;
    [FormerlySerializedAs("maxAngular")]
    public float maxAngularIntent = 1.2f;
    public float angularDeadZone = 0.01f;
    public float maxAngularAccel = 5.0f;

    [Header("Sampling")]
    [FormerlySerializedAs("minDeltaTime")]
    public float minSampleDeltaTime = 0.005f;

    [Header("Debug")]
    public bool verboseLog = false;

    private readonly HandState leftHand = new HandState();
    private readonly HandState rightHand = new HandState();
    private InputAction leftPrimaryButtonAction;
    private InputAction rightPrimaryButtonAction;

    private void Awake()
    {
        if (ros == null)
            ros = ROSConnection.GetOrCreateInstance();

        leftPrimaryButtonAction = new InputAction(
            name: "Left Primary Button",
            type: InputActionType.Button,
            binding: "<XRController>{LeftHand}/primaryButton"
        );
        rightPrimaryButtonAction = new InputAction(
            name: "Right Primary Button",
            type: InputActionType.Button,
            binding: "<XRController>{RightHand}/primaryButton"
        );

        Time.fixedDeltaTime = 0.05f;   // 20Hz
    }

    private void Start()
    {
        ros.RegisterPublisher<TwistStampedMsg>(leftTopic);
        ros.RegisterPublisher<TwistStampedMsg>(rightTopic);

        InitializeHandState(leftController, leftHand);
        InitializeHandState(rightController, rightHand);

        EnableAction(leftTriggerAction);
        EnableAction(rightTriggerAction);

        Debug.Log("DualXRServo_ReferenceFrameIncremental started.");
    }

    private void OnEnable()
    {
        EnableAction(leftTriggerAction);
        EnableAction(rightTriggerAction);
        leftPrimaryButtonAction?.Enable();
        rightPrimaryButtonAction?.Enable();
    }

    private void OnDisable()
    {
        DisableAction(leftTriggerAction);
        DisableAction(rightTriggerAction);
        leftPrimaryButtonAction?.Disable();
        rightPrimaryButtonAction?.Disable();
    }

    private void Update()
    {
        SampleHand(leftController, leftHand, enableLeft, leftTriggerAction, "Left");
        SampleHand(rightController, rightHand, enableRight, rightTriggerAction, "Right");
    }

    private void FixedUpdate()
    {
        if (ros == null)
            return;

        PublishHand(leftHand, leftTopic, "Left");
        PublishHand(rightHand, rightTopic, "Right");
    }

    private void InitializeHandState(Transform ctrl, HandState hand)
    {
        hand.currentLinearRef = Vector3.zero;
        hand.currentAngularRef = Vector3.zero;
        hand.targetLinearRef = Vector3.zero;
        hand.targetAngularRef = Vector3.zero;
        hand.sampledActive = false;
        hand.wasActiveLastFrame = false;

        if (ctrl == null)
        {
            hand.hasSample = false;
            return;
        }

        hand.lastSampleRefPos = GetPositionInReference(ctrl);
        hand.lastSampleRefRot = GetRotationInReference(ctrl);
        hand.lastSampleTime = Time.unscaledTime;
        hand.hasSample = true;
    }

    private void SampleHand(
        Transform ctrl,
        HandState hand,
        bool enabled,
        InputActionProperty triggerAction,
        string handName)
    {
        if (ctrl == null)
        {
            hand.hasSample = false;
            hand.sampledActive = false;
            hand.targetLinearRef = Vector3.zero;
            hand.targetAngularRef = Vector3.zero;
            return;
        }

        Vector3 currentRefPos = GetPositionInReference(ctrl);
        Quaternion currentRefRot = GetRotationInReference(ctrl);
        float sampleTime = Time.unscaledTime;
        bool triggerPressed = ReadEnablePressed(triggerAction, triggerThreshold, handName);
        bool active = enabled && triggerPressed;

        if (!hand.hasSample)
        {
            hand.lastSampleRefPos = currentRefPos;
            hand.lastSampleRefRot = currentRefRot;
            hand.lastSampleTime = sampleTime;
            hand.hasSample = true;
            hand.sampledActive = active;
            hand.targetLinearRef = Vector3.zero;
            hand.targetAngularRef = Vector3.zero;
            return;
        }

        float sampleDt = Mathf.Max(sampleTime - hand.lastSampleTime, minSampleDeltaTime);
        Vector3 rawLinearVelRef = (currentRefPos - hand.lastSampleRefPos) / sampleDt;

        Quaternion dq = currentRefRot * Quaternion.Inverse(hand.lastSampleRefRot);
        if (dq.w < 0f)
        {
            dq.x = -dq.x;
            dq.y = -dq.y;
            dq.z = -dq.z;
            dq.w = -dq.w;
        }

        Vector3 rawAngularVelRef = QuaternionLogAngularVelocity(dq, sampleDt);

        Vector3 linearIntent = Vector3.zero;
        Vector3 angularIntent = Vector3.zero;

        if (active)
        {
            linearIntent = Vector3.ClampMagnitude(rawLinearVelRef * linearScale, maxLinearIntent);
            angularIntent = Vector3.ClampMagnitude(rawAngularVelRef * angularScale, maxAngularIntent);

            if (linearIntent.magnitude < linearDeadZone)
                linearIntent = Vector3.zero;

            if (angularIntent.magnitude < angularDeadZone)
                angularIntent = Vector3.zero;
        }

        hand.lastSampleRefPos = currentRefPos;
        hand.lastSampleRefRot = currentRefRot;
        hand.lastSampleTime = sampleTime;
        hand.sampledActive = active;
        hand.targetLinearRef = linearIntent;
        hand.targetAngularRef = angularIntent;

        if (verboseLog)
        {
            Debug.Log(
                $"[{handName}] sampled | " +
                $"active={active} | " +
                $"targetLinear=({linearIntent.x:F3}, {linearIntent.y:F3}, {linearIntent.z:F3}) | " +
                $"targetAngular=({angularIntent.x:F3}, {angularIntent.y:F3}, {angularIntent.z:F3})"
            );
        }
    }

    private void PublishHand(HandState hand, string topic, string handName)
    {
        if (!hand.sampledActive)
        {
            hand.currentLinearRef = Vector3.zero;
            hand.currentAngularRef = Vector3.zero;

            if (sendZeroWhenReleased || hand.wasActiveLastFrame)
                SendZero(topic, commandFrame);

            hand.wasActiveLastFrame = false;
            return;
        }

        float publishDt = Mathf.Max(Time.fixedDeltaTime, 1e-4f);

        hand.currentLinearRef = MoveTowardsVector(
            hand.currentLinearRef,
            hand.targetLinearRef,
            maxLinearAccel * publishDt
        );
        hand.currentAngularRef = MoveTowardsVector(
            hand.currentAngularRef,
            hand.targetAngularRef,
            maxAngularAccel * publishDt
        );

        if (hand.targetLinearRef == Vector3.zero && hand.currentLinearRef.magnitude < linearDeadZone)
            hand.currentLinearRef = Vector3.zero;

        if (hand.targetAngularRef == Vector3.zero && hand.currentAngularRef.magnitude < angularDeadZone)
            hand.currentAngularRef = Vector3.zero;

        hand.wasActiveLastFrame = true;

        Vector3 rosLinear = ConvertUnityLinearVectorToRos(hand.currentLinearRef);
        Vector3 rosAngular = ConvertUnityAngularVectorToRos(hand.currentAngularRef);
        // Vector3 rosLinear = new Vector3(0, 0, 0);
        // Vector3 rosAngular =  new Vector3(0, 0, 0.3f);

        if (verboseLog)
        {
            Debug.Log(
                $"[{handName}] active | " +
                $"refLinear=({hand.currentLinearRef.x:F3}, {hand.currentLinearRef.y:F3}, {hand.currentLinearRef.z:F3}) | " +
                $"refAngular=({hand.currentAngularRef.x:F3}, {hand.currentAngularRef.y:F3}, {hand.currentAngularRef.z:F3}) | " +
                $"rosLinear=({rosLinear.x:F3}, {rosLinear.y:F3}, {rosLinear.z:F3}) | " +
                $"rosAngular=({rosAngular.x:F3}, {rosAngular.y:F3}, {rosAngular.z:F3})"
            );
        }

        ros.Publish(topic, BuildMsg(commandFrame, rosLinear, rosAngular));
    }

    private static Vector3 MoveTowardsVector(Vector3 current, Vector3 target, float maxDelta)
    {
        Vector3 delta = target - current;
        float magnitude = delta.magnitude;

        if (magnitude <= maxDelta || magnitude <= 1e-6f)
            return target;

        return current + delta / magnitude * maxDelta;
    }

    private Vector3 GetPositionInReference(Transform ctrl)
    {
        if (referenceFrame == null)
            return ctrl.position;

        return referenceFrame.InverseTransformPoint(ctrl.position);
    }

    private Quaternion GetRotationInReference(Transform ctrl)
    {
        if (referenceFrame == null)
            return ctrl.rotation;

        return Quaternion.Inverse(referenceFrame.rotation) * ctrl.rotation;
    }

    private static Vector3 QuaternionLogAngularVelocity(Quaternion dq, float dt)
    {
        if (dt <= 1e-6f)
            return Vector3.zero;

        Vector3 v = new Vector3(dq.x, dq.y, dq.z);
        float vNorm = v.magnitude;
        float w = Mathf.Clamp(dq.w, -1f, 1f);

        if (vNorm < 1e-8f)
            return Vector3.zero;

        // angle = 2 * atan2(|v|, w)
        float angle = 2f * Mathf.Atan2(vNorm, w);   // rad
        Vector3 axis = v / vNorm;                   // unit axis

        if (angle > Mathf.PI)
            angle -= 2f * Mathf.PI;

        return axis * (angle / dt);
    }

    private Vector3 ConvertUnityLinearVectorToRos(Vector3 unityVec)
    {
        return new Vector3(
            unityVec.z,
            -unityVec.x,
            unityVec.y
        );
    }

    private Vector3 ConvertUnityAngularVectorToRos(Vector3 unityVec)
    {
        return new Vector3(
            -unityVec.z,
            unityVec.x,
            -unityVec.y
        );
    }

    private TwistStampedMsg BuildMsg(string frameId, Vector3 lin, Vector3 ang)
    {
        TwistStampedMsg msg = new TwistStampedMsg();
        msg.header = new HeaderMsg();
        msg.header.frame_id = frameId;
        msg.header.stamp = new TimeMsg(0, 0);

        msg.twist = new TwistMsg();
        msg.twist.linear = new Vector3Msg(lin.x, lin.y, lin.z);
        msg.twist.angular = new Vector3Msg(ang.x, ang.y, ang.z);

        return msg;
    }

    private void SendZero(string topic, string frameId)
    {
        ros.Publish(topic, BuildMsg(frameId, Vector3.zero, Vector3.zero));
    }

    private static void EnableAction(InputActionProperty property)
    {
        if (property.action != null)
            property.action.Enable();
    }

    private static void DisableAction(InputActionProperty property)
    {
        if (property.action != null)
            property.action.Disable();
    }

    private bool ReadEnablePressed(InputActionProperty property, float threshold, string handName)
    {
        if (usePrimaryButtonsForEnable)
        {
            InputAction primaryAction = handName == "Left"
                ? leftPrimaryButtonAction
                : rightPrimaryButtonAction;

            if (primaryAction != null)
                return primaryAction.IsPressed();
        }

        return ReadTriggerPressed(property, threshold);
    }

    private static bool ReadTriggerPressed(InputActionProperty property, float threshold)
    {
        if (property.action == null)
            return false;

        try
        {
            if (property.action.IsPressed())
                return true;

            float value = property.action.ReadValue<float>();
            return value > threshold;
        }
        catch
        {
            return false;
        }
    }
}





