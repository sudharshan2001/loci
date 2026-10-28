# Eval results (questions.yaml)

- questions: 18
- accuracy: 61%
- refusals correct: 3/3
- verified on first attempt: 67%
- median latency: 35.9 s
- cloud calls: 0 (local only)

| | question | answer | why | s |
| --- | --- | --- | --- | --- |
| ❌ | What's on my desk? | I'm not sure. The last confirmed sighting of the desk was in the hallway, at 0:18. | not sure (unverified) | 51.5 |
| ✅ | Where is my laptop? | I last saw a laptop on the desk in the bedroom, at 0:06. It is also on the desk in the office, last seen at 0:06. | labels ['laptop', 'laptop', 'laptop', 'laptop', 'laptop'] | 36.0 |
| ✅ | Where's the keyboard? | The keyboard is on the desk in the office, last seen at 0:06. | labels ['keyboard', 'keyboard', 'keyboard', 'keyboard', 'keyboard', 'keyboard', 'keyboard'] | 15.0 |
| ✅ | Is the laundry done? | I haven't seen a washing machine or dryer in any capture. The only laundry-related thing I saw is a laundry pile left of the desk in the hallway at 0:11, with laundry buckets nearby. There's no evidence the laundry is done. | labels ['laundry pile', 'laundry bucket', 'laundry bucket', 'laundry bucket', 'laundry bucket', 'laundry bucket', 'laundry bucket', 'laundry bucket', 'clothes pile'] | 37.0 |
| ✅ | Where's the purple bucket? | I’ve seen a purple laundry bucket in the hallway left of the desk, last seen at 0:11. | labels ['laundry bucket'] | 17.3 |
| ✅ | Where did I leave my backpack? | You left a backpack right of the desk in the hallway, last seen at 0:18. | labels ['backpack', 'backpack', 'backpack', 'backpack', 'backpack'] | 36.6 |
| ❌ | Where's the laptop stand? | I haven't seen a laptop stand in any capture. | refused | 25.3 |
| ✅ | Is the stove on? | Yes, the gas stove was on as of 0:15. The right burner was lit with nothing on it when it was last seen at 0:16 in the kitchen. | no expectation (manual check) | 10.1 |
| ❌ | Are the lights on? | I haven't seen a light in any capture. | refused | 12.5 |
| ✅ | How far is the desk from the door? | The desk in the office is about 0.9 m from the office door as of 0:06. | no expectation (manual check) | 35.7 |
| ✅ | Where's my passport? | I haven't seen a passport in any capture. | refused | 9.6 |
| ✅ | Where are my car keys? | I haven't seen a car keys in any capture. | refused | 5.8 |
| ❌ | Where was the coconut stall? | I haven't seen a coconut stall in any capture. | refused | 66.8 |
| ❌ | What shops did I pass? | I'm not sure. The last confirmed sighting of the signboard was on your right, about 62 m into the walk, at 0:22. | not sure (unverified) | 94.3 |
| ❌ | Was the coconut stall open? | I haven't seen a coconut stall in any capture. | refused | 175.9 |
| ❌ | How do I get from the coconut stall to the next shop? | I haven't seen a coconut stall in any capture. | refused | 54.4 |
| ✅ | Which side of the road were the shops on? | In the street walk the shop fronts you passed were on your right, e.g. the shop front first seen at 0:09 on your right about 33 m into the walk, passed at 0:13; another shop front on your right about 37 m in, passed at 0:14. | labels ['shop front', 'shop front', 'shop front'] | 49.6 |
| ✅ | Where's the nearest ATM? | I haven't seen an ATM in any capture. | refused | 8.9 |
