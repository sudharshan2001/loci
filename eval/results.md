# Eval results (questions.yaml)

- questions: 23
- accuracy: 74%
- refusals correct: 3/3
- verified on first attempt: 74%
- median latency: 17.9 s
- cloud calls: 0 (local only)

| | question | answer | why | s |
| --- | --- | --- | --- | --- |
| ✅ | What's on my desk? | The most recent desk sighting at 0:18 is the hallway desk, which the capture shows in the hallway with no items listed as on it in the tools. The office desk that was last seen at 0:06 is in the office and, when you left it, it was holding a computer monitor, a keyboard, a computer mouse, another keyboard, a laptop, another laptop, another keyboard, another keyboard, another keyboard, another laptop, another mouse, another monitor, plus a kitchen shelf and a pillow that were also recorded on it in the hallway context. | labels ['desk', 'computer monitor', 'keyboard', 'computer mouse', 'keyboard', 'laptop', 'laptop', 'keyboard', 'keyboard', 'keyboard', 'laptop', 'computer mouse', 'computer monitor'] | 41.9 |
| ✅ | Where is my laptop? | Your laptop was last seen on the desk in the bedroom at 0:06; the newest sighting is on the desk in the bedroom from the home walk. | labels ['laptop', 'laptop', 'laptop'] | 12.0 |
| ✅ | Where's the keyboard? | The keyboard is on the desk in the office, last seen at 0:06 when you left the office. | labels ['keyboard', 'keyboard', 'keyboard'] | 12.5 |
| ❌ | Is the laundry done? | I'm not sure. The last confirmed sighting of the laundry bucket was left of the desk, in the hallway, at 0:18. | not sure (unverified) | 31.9 |
| ✅ | Where's the purple bucket? | The purple laundry bucket is left of the desk, in the hallway, last seen at 0:11 when you were in the hallway. | labels ['laundry bucket', 'laundry bucket', 'desk'] | 13.4 |
| ✅ | Where did I leave my backpack? | I last saw your backpack right of the desk, in the hallway at 0:18. It moved between sightings in that capture, so check by the desk in the hallway. | labels ['backpack', 'backpack', 'backpack', 'backpack'] | 13.3 |
| ❌ | Where's the laptop stand? | I haven't seen a laptop stand in any capture. | refused | 21.0 |
| ✅ | Is the stove on? | The gas stove was on as of 0:15 — the right burner was lit with nothing on it. This is when it was last seen at 0:16, so state is as of 0:15. | no expectation (manual check) | 6.9 |
| ❌ | Are the lights on? | I haven't seen any lights in any capture. | refused | 11.2 |
| ✅ | How far is the desk from the door? | The desk you saw in the hallway is about 2.6 m straight-line from the bedroom door you saw at 0:13. The desk is in the hallway, the door is in the bedroom. | no expectation (manual check) | 22.6 |
| ✅ | Where's my passport? | I haven't seen a passport in any capture. | refused | 10.8 |
| ✅ | Where are my car keys? | I haven't seen car keys in any capture. | refused | 9.1 |
| ✅ | ಸ್ಟವ್ ಆನ್ ಆಗಿತ್ತಾ? | ಹೌದು, ಅಡುಗೆಮನೆಯಲ್ಲಿರುವ ಗ್ಯಾಸ್ ಸ್ಟವ್ 0:15ರಲ್ಲಿ ಬಲ ಬರ್ನರ್ ಬೆಳಗಿತ್ತು. | labels ['gas stove', 'gas stove'] | 12.0 |
| ✅ | அடுப்பு எரிந்துகொண்டிருந்ததா? | ஆம், அடுப்பை நீங்கள் விட்டுச்சென்றபோது 0:15 மணிக்கு எரிந்துகொண்டிருந்தது — வலது பர்னர் எரிந்து, அதில் எதுவும் வைக்கப்படவில்லை. | labels ['gas stove'] | 12.0 |
| ✅ | என் பை எங்கே? | கடைசியாக பார்த்தது 0:18-ல் ஹால்வேயில் மேசையில் (மேசையில், ஹால்வேயில்) இருந்த பிளாஸ்டிக் பையாகும். அது நகரக்கூடியது, அப்போது மேசையில் இருந்தது; இப்போது இருக்கிறதா என்பது உறுதியில்லை. | labels ['plastic bag', 'plastic bag', 'plastic bag', 'plastic bag'] | 17.9 |
| ✅ | Where was the coconut stall? | The coconut stall is on your left, about 12 m into the walk, last seen at 0:02. | labels ['fruit stand', 'fruit stand'] | 63.2 |
| ❌ | What shops did I pass? | I'm not sure, and I couldn't find it in any capture. | refused | 56.8 |
| ✅ | Was the coconut stall open? | I haven't seen a clearly labelled coconut stall on this walk. The only coconut selling cart that appears in the captures is a cart on your left about 12 m into the walk, seen at 0:02, and it is a coconut selling cart but there is no sign of being actively operated/open. | labels ['fruit stand'] | 118.4 |
| ❌ | How do I get from the coconut stall to the next shop? | I haven't seen a coconut stall in any capture. | refused | 54.2 |
| ✅ | Which side of the road were the shops on? | Shops were seen on both sides of the road during the walk. On your right, a shop front was on your right about 33 m into the walk when it was seen at 0:09, and the same side continued with more right-side shops later; left-side shops/signs were also passed, e.g. a shop sign SH 7 on your left about 44 m into the walk at 0:11. | labels ['shop front', 'shop front', 'shop front', 'shop sign'] | 38.6 |
| ✅ | Where's the nearest ATM? | I haven't seen an ATM in any capture. | refused | 9.1 |
| ✅ | ಶ್ರೀ ಲಕ್ಷ್ಮಿ ಅಂಗಡಿ ಎಲ್ಲಿತ್ತು? | SRI LAKSHMI ಶ್ರೀ ಲಕ್ಷ್ಮಿ ಎಂದು ಬರೆದಿರುವ signboard ನಡೆದ ವಾಕ್‌ನಲ್ಲಿ ಸುಮಾರು 62 ಮೀಟರ್ ನಲ್ಲಿ ನಿಮ್ಮ ಬಲಗಡೆ ಇತ್ತು; 0:22 ಕ್ಕೆ ಕಂಡಿತು. | labels ['signboard'] | 22.5 |
| ❌ | தேங்காய் கடை எங்கே இருந்தது? | நான் எந்த கேப்ச்சரிலும் தேங்காய் கடையை பார்க்கவில்லை. | refused | 68.1 |
