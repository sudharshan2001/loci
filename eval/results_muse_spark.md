# Eval results (questions.yaml)

- questions: 35
- accuracy: 100%
- refusals correct: 5/5
- verified on first attempt: 100%
- median latency: 7.3 s
- agent: muse-spark-1.3, Meta Model API
- cloud calls: 132 (457833 prompt, 35170 completion tokens)

| | question | answer | why | s |
| --- | --- | --- | --- | --- |
| ✅ | What's on my desk? | On your office desk you've got a laptop on a stand, a monitor, a keyboard and a pillow. | labels ['desk', 'keyboard', 'laptop', 'computer monitor', 'laptop stand', 'pillow'] | 6.9 |
| ✅ | Where is my laptop? | Your laptop was in the bedroom on your Oct 29 walk at 0:24 — that was a while ago and laptops move, so check there first. | labels ['laptop'] | 7.3 |
| ✅ | Where's the keyboard? | It was in the bedroom on your Oct 29 walk at 0:23. | labels ['keyboard'] | 8.4 |
| ✅ | Is the laundry done? | Not yet — as of 0:21 on your Oct 29 walk there was still a laundry pile and a full bucket in the bedroom. | labels ['laundry pile', 'laundry bucket'] | 5.7 |
| ✅ | Where's the purple bucket? | It was left of the desk in the bedroom on your Oct 28 walk — that was a while ago and buckets move, so check there. | labels ['laundry bucket'] | 8.4 |
| ✅ | Where did I leave my backpack? | You left it left of the desk in the bedroom, seen at 0:18 on your Oct 28 walk — but that was a while ago and backpacks move, so check there. | labels ['backpack'] | 6.9 |
| ✅ | Where's the laptop stand? | On the desk in the office, as of your Oct 28 walk — that was a few days ago and it can move, so have a look there. | labels ['laptop stand'] | 7.3 |
| ✅ | Is the stove on? | No — the burner was not lit as of 0:00 on your Oct 29 walk. It was lit on your Oct 28 walk. | no expectation (manual check) | 6.8 |
| ✅ | Are the lights on? | I haven't seen whether the lights are on in any capture. | refused | 11.0 |
| ✅ | How far is the desk from the door? | The desk in the office is about a metre from the door. | no expectation (manual check) | 9.4 |
| ✅ | Where's my passport? | I haven't seen a passport in any capture. | refused | 6.7 |
| ✅ | Where are my car keys? | I haven't seen car keys in any capture. | refused | 7.1 |
| ✅ | ಸ್ಟವ್ ಆನ್ ಆಗಿತ್ತಾ? | ಇಲ್ಲ, ನೀವು ಅಕ್ಟೋಬರ್ 29 ರ ವಾಕ್‌ನಲ್ಲಿ ಅಡುಗೆಮನೆಯನ್ನು 0:00 ಕ್ಕೆ ನೋಡಿದಾಗ ಸ್ಟವ್ ಆನ್ ಇರಲಿಲ್ಲ. ಅಕ್ಟೋಬರ್ 28 ರ ವಾಕ್‌ನಲ್ಲಿ ಅದು ಉರಿಯುತ್ತಿತ್ತು. | labels ['gas stove'] | 6.1 |
| ✅ | அடுப்பு எரிந்துகொண்டிருந்ததா? | அக்ட் 29 நடையில் 0:00-ல் சமையலறையை பார்த்தபோது அடுப்பு எரியவில்லை, ஆனால் அக்ட் 28 நடையில் வலது பர்னர் எரிந்துகொண்டிருந்தது. | labels ['gas stove'] | 6.7 |
| ✅ | என் பை எங்கே? | உங்கள் பை அக் 29 நடையில் 0:23-ல் படுக்கையறையில் இருந்தது, அது கொஞ்சம் பழைய காட்சி, பைகள் நகரும் அதனால் சரிபாருங்கள். | labels ['plastic bag'] | 7.8 |
| ✅ | What changed in my home since the first capture? | Your right gas burner was lit at 0:15 on your Oct 28 walk, and it was off as of 0:00 on your Oct 29 walk. Nothing moved; the air fryer and cardboard box were only seen on your Oct 29 walk, while the backpack and laptop stand were only seen on your Oct 28 walk, but they may just not have been filmed. | no expectation (manual check) | 9.0 |
| ✅ | Where was the coconut stall? | The coconut stall was on your left about 12 metres into your street walk. | labels ['fruit stand'] | 10.1 |
| ✅ | What shops did I pass? | On your Oct 29 walk from your office you passed Reebok on your left at 21m, FOREVER 21 on your left at 23m, VIP on your right at 60m, J.C.CHANDRA JEWELLERS on your right at 64m, and megamart on your left at 94m, among many others. | labels ['signboard', 'signboard', 'signboard', 'signboard', 'signboard'] | 10.3 |
| ✅ | Was the coconut stall open? | Yes — the coconut stall on your left about 12 m into your Oct 28 walk was open, as of 0:00 into that walk. | labels ['fruit stand'] | 6.6 |
| ✅ | How do I get from the coconut stall to the next shop? | From the coconut stall keep walking straight about 20 metres, and the next shop is on your right, about 33 m into the walk. | no expectation (manual check) | 17.5 |
| ✅ | Which side of the road were the shops on? | On your newest walk most shops were on your left — Reebok at 21 metres, FOREVER 21, then BLUESTONE and BHIMA — with just a couple on your right like VIP at 60 metres. | labels ['signboard', 'signboard', 'signboard', 'signboard', 'signboard'] | 6.4 |
| ✅ | Where's the nearest ATM? | I haven't seen an ATM in any capture. The closest bank I saw was Karnataka Bank on your left about 84 m into your walk. | refused | 29.8 |
| ✅ | ಶ್ರೀ ಲಕ್ಷ್ಮಿ ಅಂಗಡಿ ಎಲ್ಲಿತ್ತು? | SRI LAKSHMI ಶ್ರೀ ಲಕ್ಷ್ಮಿ ಅಂಗಡಿ ನಿಮ್ಮ ಬಲಭಾಗದಲ್ಲಿ, ನಡಿಗೆಯಲ್ಲಿ ಸುಮಾರು 62 ಮೀಟರ್ ದೂರದಲ್ಲಿ 0:22 ಕ್ಕೆ ಕಾಣಿಸಿತು. | labels ['shop sign'] | 5.4 |
| ✅ | தேங்காய் கடை எங்கே இருந்தது? | தேங்காய் கடை உங்கள் நடையின் தொடக்கத்தில் இடது பக்கம், சுமார் 12 மீட்டரில் இருந்தது. | labels ['fruit stand'] | 6.8 |
| ✅ | Where was Kalyan Jewellers? | KALYAN was on your left, about 144 m into the walk, passed at 2:21. | labels ['signboard'] | 8.2 |
| ✅ | Where was the Levi's store? | The Levi's OUTLET was on your left, about 313 m into your Oct 29 walk from your office, at 5:09. | labels ['shop sign'] | 7.0 |
| ✅ | Which side of the road was BIBA on? | BIBA was on your left, about 187 metres into the walk at around 3:01. | labels ['signboard', 'shop front'] | 6.5 |
| ✅ | How far from my office was Louis Philippe? | Louis Philippe Outlet was about 204 metres from your office, on your left about 3:25 into your Oct 29 walk. | labels ['signboard'] | 9.7 |
| ✅ | How do I get from Kalyan Jewellers to Levi's? | Keep going straight about 170 metres the way you were walking, Levi's Outlet is on your left past GRT Jewellers and Louis Philippe. | no expectation (manual check) | 6.9 |
| ✅ | Did I pass a fruit cart? | Yes, you passed a blue fruit cart on your right about 50 metres into your Oct 29 walk from your office. | labels ['fruit stall'] | 8.2 |
| ✅ | What clothing stores did I pass? | On your newest walk you passed Reebok on the left at 21m, FOREVER 21 on the left at 23m, Soch on the left at 130m, BIBA on the left at 187m, LOUIS PHILIPPE OUTLET on the left at 204m, Van Heusen on the left at 246m, Manyavar on the left at 257m, Levi's OUTLET on the left at 313m, RED TAPE on the left at 355m and BRAND FACTORY on the left at 368m. | labels ['signboard', 'signboard', 'shop sign', 'signboard', 'signboard', 'signboard', 'signboard', 'shop sign', 'signboard', 'signboard'] | 7.9 |
| ✅ | Was there a Lenovo shop? | Yes, there was a Lenovo sign on your left about 278 m into your walk, at 4:41. | labels ['signboard'] | 5.0 |
| ✅ | Where's the nearest metro station? | I haven't seen a metro station in any capture. | refused | 7.4 |
| ✅ | ಕಲ್ಯಾಣ್ ಜ್ಯುವೆಲ್ಲರ್ಸ್ ಎಲ್ಲಿತ್ತು? | KALYAN ಅಂಗಡಿ ನಿಮ್ಮ ಎಡಭಾಗದಲ್ಲಿ, ನಡಿಗೆಯಲ್ಲಿ ಸುಮಾರು 144 ಮೀಟರ್ ಒಳಗೆ ಇತ್ತು. | labels ['signboard'] | 7.4 |
| ✅ | லீவைஸ் கடை எங்கே இருந்தது? | Levi's OUTLET கடை உங்கள் நடையில் இடது பக்கம், சுமார் 313 மீட்டர் தொலைவில் இருந்தது. | labels ['shop sign'] | 5.7 |
