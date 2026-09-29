#!/usr/bin/env python3
"""Generate 100 unique part-semantic prompts for the expanded benchmark."""
import json

PARTS = ["torso", "left_arm", "right_arm", "left_leg", "right_leg"]

PROMPTS = {
    "torso": [
        "a person bends the torso forward",
        "a person twists the upper body to the left",
        "a person leans backward while standing",
        "a person rotates the torso from side to side",
        "a person turns their body around to the left",
        "a person bends their torso down forward",
        "a person twists their upper body sideways",
        "a person leans their torso to the right",
        "a person rotates their body slowly",
        "a person leans their torso to the left",
        "a person bends sideways to the right",
        "a person bends sideways to the left",
        "a person shrugs their shoulders",
        "a person hunches their back forward",
        "a person arches their back backward",
        "a person rotates their hips in a circle",
        "a person leans the torso forward while standing",
        "a person twists the torso and looks over their shoulder",
        "a person bends forward and touches the ground",
        "a person rotates the upper body while keeping feet planted",
    ],
    "left_arm": [
        "a person raises their left arm up above the head",
        "a person waves their left hand in the air",
        "a person lifts their left arm sideways",
        "a person stretches their left arm forward",
        "a person puts their left hand on the hip",
        "a person extends the left arm to the side",
        "a person swings the left arm forward",
        "a person raises the left hand to the face",
        "a person crosses the left arm over the chest",
        "a person bends the left elbow and raises the forearm",
        "a person reaches up with the left hand",
        "a person pushes forward with the left hand",
        "a person pulls the left arm back",
        "a person circles the left arm around",
        "a person lowers the left arm slowly",
        "a person holds the left arm out straight",
        "a person raises the left arm while stepping forward",
        "a person waves the left hand and turns around",
        "a person extends the left arm and leans sideways",
        "a person raises the left arm above the head while walking",
    ],
    "right_arm": [
        "a person raises their right arm up above the head",
        "a person waves their right hand in the air",
        "a person lifts their right arm sideways",
        "a person stretches their right arm forward",
        "a person puts their right hand on the hip",
        "a person extends the right arm to the side",
        "a person swings the right arm forward",
        "a person raises the right hand to the face",
        "a person crosses the right arm over the chest",
        "a person bends the right elbow and raises the forearm",
        "a person reaches up with the right hand",
        "a person pushes forward with the right hand",
        "a person pulls the right arm back",
        "a person circles the right arm around",
        "a person lowers the right arm slowly",
        "a person holds the right arm out straight",
        "a person raises the right arm while stepping forward",
        "a person waves the right hand and turns around",
        "a person extends the right arm and leans sideways",
        "a person raises the right arm above the head while walking",
    ],
    "left_leg": [
        "a person kicks forward with their left leg",
        "a person lifts their left knee up",
        "a person steps forward with their left leg",
        "a person raises their left foot off the ground",
        "a person bends their left knee",
        "a person kicks sideways with the left leg",
        "a person steps sideways with the left foot",
        "a person lifts the left foot to the side",
        "a person extends the left leg backward",
        "a person stomps the left foot on the ground",
        "a person raises the left knee to the chest",
        "a person kicks backward with the left leg",
        "a person taps the left toe on the ground",
        "a person swings the left leg forward",
        "a person lifts the left heel while standing",
        "a person steps backward with the left foot",
        "a person kicks the left leg while raising the arms",
        "a person steps forward with the left foot and turns",
        "a person lifts the left knee while bending forward",
        "a person kicks sideways with the left leg while standing",
    ],
    "right_leg": [
        "a person kicks forward with their right leg",
        "a person lifts their right knee up",
        "a person steps forward with their right leg",
        "a person raises their right foot off the ground",
        "a person bends their right knee",
        "a person kicks sideways with the right leg",
        "a person steps sideways with the right foot",
        "a person lifts the right foot to the side",
        "a person extends the right leg backward",
        "a person stomps the right foot on the ground",
        "a person raises the right knee to the chest",
        "a person kicks backward with the right leg",
        "a person taps the right toe on the ground",
        "a person swings the right leg forward",
        "a person lifts the right heel while standing",
        "a person steps backward with the right foot",
        "a person kicks the right leg while raising the arms",
        "a person steps forward with the right foot and turns",
        "a person lifts the right knee while bending forward",
        "a person kicks sideways with the right leg while standing",
    ],
}

def main():
    entries = []
    for part in PARTS:
        prompts = PROMPTS[part]
        for i, text in enumerate(prompts):
            difficulty = "compound" if i >= 16 else "simple"
            category = "compound" if i >= 16 else "simple"
            entries.append({
                "id": f"{part}_{i+1:02d}",
                "text": text,
                "target_part": part,
                "category": category,
                "difficulty": difficulty,
            })

    out = "evaluation/prompts/part_semantic_v2.json"
    with open(out, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"Wrote {len(entries)} prompts to {out}")
    # Print distribution
    from collections import Counter
    dist = Counter(e["target_part"] for e in entries)
    for p in PARTS:
        print(f"  {p}: {dist[p]}")
    # Left/right pairs
    lr = sum(1 for e in entries if "left" in e["text"] or "right" in e["text"])
    print(f"  left/right mentions: {lr}")

if __name__ == "__main__":
    main()
