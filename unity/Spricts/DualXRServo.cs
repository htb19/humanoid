using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Controls;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;
using RosMessageTypes.Std;
using RosMessageTypes.BuiltinInterfaces;

public class DualXRServo_LocalFrame_TriggerEnabled : MonoBehaviour
{
    [Header("ROS")]
    public ROSConnection ros;
    public string leftTopic = "/unity/left_twist_raw";
    public string rightTopic = "/unity/right_twist_raw";
    public string leftEeFrame = "left_wrist_yaw_link";
    public string rightEeFrame = "right_wrist_yaw_link";

    [Header("Controller Transforms")]
    public Transform leftController;
    public Transform rightController;

    [Header("Global Enable")]
    public bool enableLeft = true;
    public bool enableRight = true;

    [Header("Trigger-to-Enable")]
    [Tooltip("按住左手扳机时才发送左手速度")]
    public InputActionProperty leftTriggerAction;

    [Tooltip("按住右手扳机时才发送右手速度")]
    public InputActionProperty rightTriggerAction;

    [Range(0f, 1f)]
    [Tooltip("扳机值大于该阈值时视为使能")]
    public float triggerThreshold = 0.15f;

    [Tooltip("是否在未按扳机时持续发送零速度")]
    public bool sendZeroWhenReleased = false;

    [Header("Linear Control")]
    public float linearScale = 2.0f;
    public float maxLinear =5.0f;
    public float linearDeadZone = 0.2f;

    [Header("Angular Control")]
    public float angularScale = 2.0f;
    public float maxAngular = 5.0f;
    public float angularDeadZone = 0.2f;

    [Header("Smoothing")]
    [Range(0.01f, 1.0f)]
    public float smoothFactor = 0.2f;
    public float minDeltaTime = 0.01f;

    [Header("Debug")]
    public bool verboseLog = false;

    private Vector3 leftLastWorldPos;
    private Quaternion leftLastWorldRot;
    private Vector3 leftSmoothVel = Vector3.zero;
    private Vector3 leftSmoothAngularVel = Vector3.zero;
    private bool leftWasActiveLastFrame = false;

    private Vector3 rightLastWorldPos;
    private Quaternion rightLastWorldRot;
    private Vector3 rightSmoothVel = Vector3.zero;
    private Vector3 rightSmoothAngularVel = Vector3.zero;
    private bool rightWasActiveLastFrame = false;

    private void Start()
    {
        if (ros == null)
            ros = ROSConnection.GetOrCreateInstance();

        ros.RegisterPublisher<TwistStampedMsg>(leftTopic);
        ros.RegisterPublisher<TwistStampedMsg>(rightTopic);

        if (leftController != null)
        {
            leftLastWorldPos = leftController.position;
            leftLastWorldRot = leftController.rotation;
        }

        if (rightController != null)
        {
            rightLastWorldPos = rightController.position;
            rightLastWorldRot = rightController.rotation;
        }

        EnableAction(leftTriggerAction);
        EnableAction(rightTriggerAction);

        Debug.Log("DualXRServo_LocalFrame_TriggerEnabled started.");
    }

    private void OnEnable()
    {
        EnableAction(leftTriggerAction);
        EnableAction(rightTriggerAction);
    }

    private void OnDisable()
    {
        DisableAction(leftTriggerAction);
        DisableAction(rightTriggerAction);
    }

    private void Update()
    {
        if (ros == null)
            return;

        if (leftController != null)
        {
            bool leftTriggerPressed = ReadTriggerPressed(leftTriggerAction, triggerThreshold);
            bool leftActive = enableLeft && leftTriggerPressed;

            ProcessHand(
                leftController,
                ref leftLastWorldPos,
                ref leftLastWorldRot,
                ref leftSmoothVel,
                ref leftSmoothAngularVel,
                leftTopic,
                leftEeFrame,
                leftActive,
                ref leftWasActiveLastFrame,
                "Left"
            );
        }

        if (rightController != null)
        {
            bool rightTriggerPressed = ReadTriggerPressed(rightTriggerAction, triggerThreshold);
            bool rightActive = enableRight && rightTriggerPressed;

            ProcessHand(
                rightController,
                ref rightLastWorldPos,
                ref rightLastWorldRot,
                ref rightSmoothVel,
                ref rightSmoothAngularVel,
                rightTopic,
                rightEeFrame,
                rightActive,
                ref rightWasActiveLastFrame,
                "Right"
            );
        }
    }

    private void ProcessHand(
        Transform ctrl,
        ref Vector3 lastWorldPos,
        ref Quaternion lastWorldRot,
        ref Vector3 smoothVel,
        ref Vector3 smoothAngularVel,
        string topic,
        string eeFrame,
        bool active,
        ref bool wasActiveLastFrame,
        string handName)
    {
        if (!active)
        {
            ResetSingle(ctrl, ref lastWorldPos, ref lastWorldRot);
            smoothVel = Vector3.zero;
            smoothAngularVel = Vector3.zero;

            if (sendZeroWhenReleased || wasActiveLastFrame)
                SendZero(topic, eeFrame);

            wasActiveLastFrame = false;
            return;
        }

        float dt = Mathf.Max(Time.deltaTime, minDeltaTime);

        Vector3 worldDelta = ctrl.position - lastWorldPos;
        Vector3 localDelta = ctrl.InverseTransformDirection(worldDelta);
        Vector3 rawLinearVel = localDelta / dt;

        smoothVel = Vector3.Lerp(smoothVel, rawLinearVel, smoothFactor);
        Vector3 linear = Vector3.ClampMagnitude(smoothVel * linearScale, maxLinear);

        if (linear.magnitude < linearDeadZone)
        {
            linear = Vector3.zero;
            smoothVel = Vector3.zero;
        }

        Quaternion deltaWorldRot = ctrl.rotation * Quaternion.Inverse(lastWorldRot);
        deltaWorldRot.ToAngleAxis(out float angleDeg, out Vector3 axisWorld);

        if (angleDeg > 180f)
            angleDeg -= 360f;

        Vector3 rawAngularVel = Vector3.zero;

        if (Mathf.Abs(angleDeg) > 0.01f && axisWorld.sqrMagnitude > 0.0001f)
        {
            Vector3 axisLocal = ctrl.InverseTransformDirection(axisWorld.normalized);
            rawAngularVel = axisLocal * (angleDeg * Mathf.Deg2Rad / dt);
        }

        smoothAngularVel = Vector3.Lerp(smoothAngularVel, rawAngularVel, smoothFactor);
        Vector3 angular = Vector3.ClampMagnitude(smoothAngularVel * angularScale, maxAngular);

        if (angular.magnitude < angularDeadZone)
        {
            angular = Vector3.zero;
            smoothAngularVel = Vector3.zero;
        }

        lastWorldPos = ctrl.position;
        lastWorldRot = ctrl.rotation;
        wasActiveLastFrame = true;

        Vector3 rosLinear = new Vector3(linear.x, -linear.y, linear.z);
        Vector3 rosAngular = new Vector3(angular.x, -angular.y, angular.z);
        // Vector3 rosLinear = new Vector3(0.0f, 0.2f, 0.0f);
        // Vector3 rosAngular = new Vector3(0.0f, 0.0f, 0.0f);

        if (verboseLog)
        {
            Debug.Log(
                $"[{handName}][{eeFrame}] trigger active | " +
                $"linear=({rosLinear.x:F3}, {rosLinear.y:F3}, {rosLinear.z:F3}) | " +
                $"angular=({rosAngular.x:F3}, {rosAngular.y:F3}, {rosAngular.z:F3})"
            );
        }

        ros.Publish(topic, BuildMsg(eeFrame, rosLinear, rosAngular));
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

    private void SendZero(string topic, string eeFrame)
    {
        ros.Publish(topic, BuildMsg(eeFrame, Vector3.zero, Vector3.zero));
    }

    private void ResetSingle(Transform ctrl, ref Vector3 pos, ref Quaternion rot)
    {
        pos = ctrl.position;
        rot = ctrl.rotation;
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

    private static bool ReadTriggerPressed(InputActionProperty property, float threshold)
    {
        if (property.action == null)
            return false;

        try
        {
            float value = property.action.ReadValue<float>();
            return value > threshold;
        }
        catch
        {
            return false;
        }
    }
}





