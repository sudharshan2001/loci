# Eval results (questions.yaml)

- questions: 18
- accuracy: 72%
- refusals correct: 3/3
- verified on first attempt: 89%
- median latency: 22.1 s
- cloud calls: 0 (local only)

| | question | answer | why | s |
| --- | --- | --- | --- | --- |
| ❌ | What's on my desk? | I'm not sure. The last confirmed sighting of the desk was in the hallway, at 0:18. | not sure (unverified) | 65.6 |
| ✅ | Where is my laptop? | I last saw a laptop on the desk in the bedroom, at about 0:06 into the home capture. | labels ['laptop', 'laptop', 'laptop'] | 17.6 |
| ✅ | Where's the keyboard? | The keyboard you saw most clearly is on the desk in the office, last seen at 0:06. There’s also a low-confidence detection on the desk in the bedroom. | labels ['keyboard', 'keyboard', 'keyboard'] | 19.8 |
| ✅ | Is the laundry done? | I haven't seen a washing machine or dryer in any capture, and a laundry pile was left of the desk in the hallway as of 0:11, so I can't confirm if the laundry is done. | labels ['laundry pile', 'laundry bucket', 'laundry bucket', 'laundry bucket'] | 24.1 |
| ✅ | Where's the purple bucket? | The purple laundry bucket was last seen in the hallway left of the desk at 0:11. | labels ['laundry bucket', 'laundry bucket', 'laundry bucket', 'laundry bucket'] | 39.8 |
| ✅ | Where did I leave my backpack? | I last saw a backpack in the hallway right of the desk, in the hallway, at 0:18. That’s the most recent sighting in the home capture. | labels ['backpack', 'backpack', 'backpack'] | 27.8 |
| ❌ | Where's the laptop stand? | I haven't seen a laptop stand in any capture. | refused | 20.0 |
| ✅ | Is the stove on? | Yes — as of 0:15 the gas stove’s right burner is lit. It was seen in the kitchen at 0:16. | no expectation (manual check) | 9.1 |
| ❌ | Are the lights on? | I haven't seen a light in any capture. | refused | 16.3 |
| ✅ | How far is the desk from the door? | The desk you’re seeing in the hallway is about 1.4 m straight-line from the door in the office — roughly 1.36 m as measured when the desk was last seen at 0:18. | no expectation (manual check) | 38.2 |
| ✅ | Where's my passport? | I haven't seen a passport in any capture. | refused | 14.6 |
| ✅ | Where are my car keys? | I haven't seen a car keys in any capture. | refused | 12.8 |
| ❌ | Where was the coconut stall? | The coconut stall was on your left, about 12 m into the walk, seen at 0:02. | labels ['fruit stand'] | 81.0 |
| ✅ | What shops did I pass? | On the street walk you passed a fruit stand on the left at 12.3 m / 0:06, a street stall on the left at the same spot, then shop fronts and signs on the right around 32–38 m / 0:13–0:15 including MMM MIRUNDA, RED CURRY, EID CANTY, 180 CENTR, BIG GUNTY, UR CITY and HI COPY!, plus SH 7 on the left at 43.7 m / 0:16, and later stalls and signs around 57–62 m / 0:20–0:22 such as Independence, MARTY, GATE, BITE, Shree Sai, Sri Lakshmi. | labels ['fruit stand', 'fruit stand', 'street stall', 'shop front', 'fruit stand', 'shop front', 'signboard', 'signboard', 'signboard', 'shop sign', 'shop sign', 'shop sign', 'signboard', 'shop sign', 'shop front', 'signboard', 'street stall', 'signboard', 'signboard', 'signboard', 'signboard', 'signboard', 'signboard', 'street stall', 'street stall', 'street stall', 'street stall', 'street stall'] | 18.3 |
| ❌ | Was the coconut stall open? | I haven't seen evidence that the coconut vendor stall was open or closed. I did see a red cart loaded with green coconuts typical of a street coconut vendor stall on your left, about 12 m into the walk, as of 0:02. | labels ['fruit stand'] | 70.3 |
| ✅ | How do I get from the coconut stall to the next shop? | From the coconut vendor cart on your left about 12 m into the walk, seen at 0:02, keep walking forward along the street about 21 m further to the next shop front on your right around 33 m into the walk, seen at 0:09. | no expectation (manual check) | 75.8 |
| ✅ | Which side of the road were the shops on? | The shops you passed were on your right side of the walk. For example, shop front about 33 m into the walk was on your right and was first seen at 0:09. | labels ['shop front', 'shop front', 'shop front'] | 44.4 |
| ✅ | Where's the nearest ATM? | I haven't seen an ATM in any capture. | refused | 12.2 |
