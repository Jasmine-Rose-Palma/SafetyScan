import cv2
from ultralytics import YOLO
from rl_agent import ThresholdRLAgent
from rl_environment import compute_reward

def run_rl_detection(model_path):
    model = YOLO(model_path)
    agent = ThresholdRLAgent()

    cap = cv2.VideoCapture(0)

    step = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # RL chooses threshold
        threshold = agent.choose_action()

        # YOLO inference
        results = model(frame, conf=threshold)[0]

        detections = []

        for box in results.boxes:
            conf = float(box.conf[0])
            detections.append({"confidence": conf})

        # Compute reward
        reward = compute_reward(detections)

        # Update agent
        agent.update(threshold, reward)
        agent.decay_epsilon()

        # Draw detections
        annotated = results.plot()

        cv2.putText(
            annotated,
            f"Threshold: {threshold:.2f} | Reward: {reward:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("PPE Detection (RL)", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        step += 1
        if step > 300:  # limit for training
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save learning curve
    agent.plot_learning()

    print("Best threshold learned:", agent.get_best_threshold())