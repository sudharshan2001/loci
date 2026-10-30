# Eval results (questions.yaml)

- questions: 34
- accuracy: 56%
- refusals correct: 4/4
- verified on first attempt: 85%
- median latency: 20.3 s
- cloud calls: 0 (local only)

| | question | answer | why | s |
| --- | --- | --- | --- | --- |
| ❌ | What's on my desk? | I'm not sure. The last confirmed sighting of the desk was in the hallway, at 0:18. | not sure (unverified) | 59.4 |
| ✅ | Where is my laptop? | Last seen at 0:23 in the bedroom, left of the desk. It’s a movable item seen yesterday, so check there. | labels ['laptop', 'laptop'] | 10.3 |
| ✅ | Where's the keyboard? | The most recent sighting is in the bedroom, left of the desk. It was seen at 0:24 in the home2 capture. | labels ['keyboard'] | 11.8 |
| ✅ | Is the laundry done? | As of 0:21 in the home2 bedroom walk, the laundry was still pending. The laundry pile in the bedroom was marked laundry_pending true at 0:21, and the laundry bucket there was also marked pending. | labels ['laundry pile', 'laundry bucket', 'clothes pile'] | 31.7 |
| ✅ | Where's the purple bucket? | The purple laundry bucket was last seen left of the desk, in the hallway at 0:11. It's movable, so it may have moved since then. | labels ['laundry bucket', 'laundry bucket', 'laundry bucket'] | 21.1 |
| ✅ | Where did I leave my backpack? | Your last sighting of the backpack was at 0:18 in the hallway, to the right of the desk. It had been moved between captures and is movable, so it’s unlikely to still be there. | labels ['backpack', 'backpack', 'backpack'] | 13.7 |
| ❌ | Where's the laptop stand? | I haven't seen a laptop stand in any capture. | refused | 22.8 |
| ✅ | Is the stove on? | In your most recent home walk (29 Oct) the gas stove in the kitchen was off when you last saw it at 0:18 — burner_lit false, nothing on a burner. As of that sighting it was not on. For reference, in the earlier 28 Oct walk the stove was on at 0:15 with the right burner lit. | no expectation (manual check) | 22.1 |
| ❌ | Are the lights on? | I haven't seen a light in any capture. | refused | 14.5 |
| ✅ | How far is the desk from the door? | The desk in the living room is about 1.3 m from the living-room door, roughly a metre-plus away as seen at 0:18. | no expectation (manual check) | 31.6 |
| ✅ | Where's my passport? | I haven't seen a passport in any capture. | refused | 7.0 |
| ✅ | Where are my car keys? | I haven't seen a car keys in any capture. | refused | 7.7 |
| ✅ | ಸ್ಟವ್ ಆನ್ ಆಗಿತ್ತಾ? | ಹೌದು, home ಕ್ಯಾಪ್ಚರ್‌ನಲ್ಲಿ 2026-10-28 16:45:46+05:30 ರಲ್ಲಿ gas stove 0:15 ಕ್ಕೆ ಕಂಡಾಗ burner_lit true ಆಗಿತ್ತು, ಬಲ ಬರ್ನರ್ ಬೆಳಗಿತ್ತು. ಇದು ಆ ಕ್ಯಾಪ್ಚರ್‌ನಲ್ಲಿ ಕಂಡ ಸ್ಥಿತಿ; evidence home:obs_0054_00. | labels ['gas stove', 'gas stove', 'gas stove', 'gas stove'] | 43.2 |
| ✅ | அடுப்பு எரிந்துகொண்டிருந்ததா? | மிகச் சமீபத்திய நடை home2-ல் அடுப்பு எரியவில்லை; 0:18-ல் பார்த்தபோது burner_lit false, எதுவும் அடுப்பில் இல்லை. | labels ['gas stove'] | 35.6 |
| ✅ | என் பை எங்கே? | உங்கள் பை (backpack) கடைசியாக ஹால்வேயில் டெஸ்கின் வலதுபுறத்தில் காணப்பட்டது. அது 0:18 இல் பார்க்கப்பட்டது, அதாவது நீங்கள் ஹால்வேயை விட்டு சென்றபோது அங்கே இருந்தது. பை நகரக்கூடியது, எனவே இப்போது அங்கே இருக்குமா என்பது உறுதி இல்லை. | labels ['backpack', 'backpack', 'backpack'] | 25.0 |
| ✅ | Where was the coconut stall? | The coconut stall is the fruit stand on your left about 12 m into the walk, seen around 0:02. | labels ['fruit stand', 'fruit stand'] | 69.1 |
| ❌ | What shops did I pass? | I'm not sure, and I couldn't find it in any capture. | refused | 60.0 |
| ✅ | Was the coconut stall open? | I haven't seen a coconut stall that was open. A coconut cart was spotted on your left about 12 m into the walk at 0:02, but it appears unmanned/not actively open. | labels ['fruit stand', 'street stall', 'street stall', 'street stall', 'fruit stand'] | 122.8 |
| ❌ | How do I get from the coconut stall to the next shop? | I haven't seen a coconut stall in any capture. | refused | 69.3 |
| ✅ | Which side of the road were the shops on? | On the street walk the shops/stalls are on both sides. Early in the walk the fruit stand and street stall are on your left (≈12 m at 0:06), and from about 26 m onward a run of shop signs and shop fronts is on the right, e.g. 180 CENTR on the right at ~26 m, 0:12. Later in the walk there are also shops on the left again, e.g. SH 7 on the left at ~43.7 m, 0:16. | labels ['shop sign', 'fruit stand', 'street stall', 'shop sign', 'shop sign'] | 14.7 |
| ✅ | Where's the nearest ATM? | I haven't seen an ATM in any capture. | refused | 10.3 |
| ✅ | ಶ್ರೀ ಲಕ್ಷ್ಮಿ ಅಂಗಡಿ ಎಲ್ಲಿತ್ತು? | ನೀವು ನಡೆದ ವಾಕ್‌ನಲ್ಲಿ ಸುಮಾರು 62 m ನಲ್ಲಿ, ಬಲಕ್ಕೆ SRI LAKSHMI ಶ್ರೀ ಲಕ್ಷ್ಮಿ ಬೋರ್ಡ್ ಕಂಡಿತು. ಅದು 0:22 ರಲ್ಲಿ, ವಾಕ್ ಆರಂಭದಿಂದ 62.3 m ದೂರದಲ್ಲಿ ಬಲಭಾಗದಲ್ಲಿತ್ತು. | labels ['signboard', 'signboard'] | 13.0 |
| ❌ | தேங்காய் கடை எங்கே இருந்தது? | நான் எந்த கேப்ச்சரிலும் தேங்காய் கடையை பார்க்கவில்லை. | refused | 56.4 |
| ❌ | Where was Kalyan Jewellers? | I haven't seen a Kalyan Jewellers in any capture. | refused | 17.2 |
| ❌ | Where was the Levi's store? | I haven't seen a Levi's store in any capture. | refused | 22.1 |
| ❌ | Which side of the road was BIBA on? | I haven't seen a BIBA in any capture. | refused | 10.3 |
| ❌ | How far from my office was Louis Philippe? | I haven't seen a Louis Philippe in any capture. | refused | 21.4 |
| ❌ | How do I get from Kalyan Jewellers to Levi's? | I haven't seen a Kalyan Jewellers or a Levi's in any capture. | refused | 19.5 |
| ✅ | Did I pass a fruit cart? | Yes, you passed a fruit cart on your left about 12 m into the walk at 0:02. | labels ['fruit stand'] | 50.9 |
| ❌ | What clothing stores did I pass? | I haven't seen a clothing store in any capture. | refused | 17.9 |
| ❌ | Was there a Lenovo shop? | I haven't seen a Lenovo shop in any capture. | refused | 7.1 |
| ✅ | Where's the nearest metro station? | I haven't seen a metro station in any capture. | refused | 12.8 |
| ❌ | ಕಲ್ಯಾಣ್ ಜ್ಯುವೆಲ್ಲರ್ಸ್ ಎಲ್ಲಿತ್ತು? | I haven't seen a Kalyan Jewellers in any capture. | refused | 12.2 |
| ❌ | லீவைஸ் கடை எங்கே இருந்தது? | நான் எந்த கேப்சரிலும் Levi's கடையை பார்க்கவில்லை. | refused | 17.2 |
