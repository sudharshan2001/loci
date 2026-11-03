# Eval results (questions.yaml)

- questions: 40
- accuracy: 98%
- refusals correct: 4/4
- verified on first attempt: 100%
- median latency: 12.8 s
- agent: muse-spark-1.3, Meta Model API
- cloud calls: 139 (469931 prompt, 37705 completion tokens)

| | question | answer | why | s |
| --- | --- | --- | --- | --- |
| ✅ | What's on my desk? | On your desk in the office there's a computer monitor, a laptop, a keyboard and a laptop stand. | labels ['desk', 'computer monitor', 'laptop', 'keyboard', 'laptop stand'] | 20.3 |
| ✅ | Where is my laptop? | Your laptop was left of the desk, seen today at 1:02 into your walk. | labels ['laptop'] | 12.5 |
| ✅ | Where's the keyboard? | It was left of the desk, seen at 1:02 on your latest walk. | labels ['keyboard'] | 10.7 |
| ✅ | Is the laundry done? | Yes — as of 0:40 on your Nov 3 walk, both laundry buckets in the bedroom had no laundry pending. There was still a pending pile on your Oct 29 walk. | labels ['laundry bucket', 'laundry bucket'] | 10.5 |
| ✅ | Where's the purple bucket? | It's in the bedroom, seen at 1:02 on your walk today. | labels ['purple bucket'] | 9.6 |
| ✅ | Where did I leave my backpack? | You left it left of the desk in the bedroom, at 0:18 on your Oct 28 walk — that was days ago and backpacks move, so check there first. | labels ['backpack'] | 13.9 |
| ✅ | Where's the laptop stand? | It was on the desk in the office at 0:06 on your Oct 28 walk, but that was a while ago and it can move, so check there first. | labels ['laptop stand'] | 24.6 |
| ✅ | Is the stove on? | No — the burner was not lit when you left the kitchen at 0:34 on Nov 3. It was lit on the right burner back on your Oct 28 walk. | no expectation (manual check) | 9.5 |
| ✅ | Are the lights on? | Yes — as of your walk today, the bedroom light was on at 0:51, and the hallway light at 0:24 and room light at 0:20 were also on. | labels ['light', 'light', 'light'] | 14.1 |
| ✅ | How far is the desk from the door? | About half a metre — the desk was 0.6 m from the door on your Nov 3 walk. | no expectation (manual check) | 26.7 |
| ✅ | Where's my passport? | I haven't seen a passport in any capture. | refused | 11.8 |
| ✅ | Where are my car keys? | I haven't seen car keys in any capture. | refused | 20.5 |
| ✅ | ಸ್ಟವ್ ಆನ್ ಆಗಿತ್ತಾ? | ಇಲ್ಲ, ನೀವು ನವೆಂಬರ್ 3 ರಂದು ಅಡುಗೆಮನೆ ಬಿಟ್ಟಾಗ 0:34 ಕ್ಕೆ ಸ್ಟವ್ ಆಫ್ ಆಗಿತ್ತು, ಆದರೆ ಅಕ್ಟೋಬರ್ 28 ರಲ್ಲಿ ಅದು ಆನ್ ಆಗಿತ್ತು. | labels ['gas stove'] | 11.1 |
| ✅ | அடுப்பு எரிந்துகொண்டிருந்ததா? | இல்லை, Nov 3 நடையில் 0:34-ல் சமையலறையை விட்டுச் செல்லும்போது அடுப்பு எரியவில்லை. Oct 28 நடையில் வலது பர்னர் எரிந்துகொண்டிருந்தது. | labels ['gas stove'] | 11.2 |
| ✅ | என் பை எங்கே? | உங்கள் பை Oct 29 நடையில் படுக்கையறையில் கடைசியாக காணப்பட்டது, அது பழைய காட்சி என்பதால் சரிபார்த்துக் கொள்ளுங்கள். | labels ['plastic bag'] | 15.1 |
| ✅ | What changed in my home since the first capture? | The gas stove burner was lit on your Oct 28 walk at 0:15, now off on your Nov 3 walk at 0:34. The laundry bucket moved in the bedroom, and a backpack and jeans were only seen on your Oct 28 walk while the blue spray bottle and notebook were only seen on your Nov 3 walk. | no expectation (manual check) | 11.6 |
| ✅ | What did I say about the plant? | You said you'll water the plants tomorrow. | labels ['potted plant'] | 15.9 |
| ✅ | Did I leave the fan on? | Yes — on your walk today two fans were on as of 0:20, while the bedroom fan was off when you left it at 0:51. | labels ['ceiling fan', 'ceiling fan', 'ceiling fan'] | 0.2 |
| ✅ | What do I have to do tomorrow? | You said you'll water the plants and change the gas cylinder tomorrow, and you also said in Tamil that you need to take out the trash tomorrow. | no expectation (manual check) | 12.2 |
| ❌ | குப்பை பற்றி நான் என்ன சொன்னேன்? | நாளைக்கு குப்பை எடுக்கணும் என்று சொன்னீர்கள். | refused | 12.9 |
| ✅ | Where's my water bottle? | Your water bottle was in the bedroom at 0:41 on your Nov 3 walk. | labels ['water bottle'] | 12.4 |
| ✅ | Where was the coconut stall? | It was on your left about 12 metres into the walk, right at the start. | labels ['fruit stand'] | 12.7 |
| ✅ | What shops did I pass? | On your Oct 29 walk from your office you passed Reebok on your left at 21m at 0:20, FOREVER 21 on your left at 23m at 0:21, VIP on your right at 60m at 0:52, J.C. Chandra Jewellers on your right at 64m at 0:57, BLUESTONE on your left at 78m at 1:18 and BHIMA on your left at 84m at 1:23, with many more further on. | labels ['signboard', 'signboard', 'signboard', 'signboard', 'signboard', 'signboard'] | 0.0 |
| ✅ | Was the coconut stall open? | Yes, it was open when you passed it on your Oct 28 walk, on your left about 12 m in. | labels ['fruit stand'] | 36.4 |
| ✅ | How do I get from the coconut stall to the next shop? | Keep walking forward about 15 metres to UR CITY, on your right about 28 m into the walk, just past 180 CENTR. | no expectation (manual check) | 24.5 |
| ✅ | Which side of the road were the shops on? | Shops were on both sides on your newest walk, mostly on your left — Reebok on the left at 21 metres, VIP on the right at 60 metres, BLUESTONE on the left at 78 metres, and W Wildcraft on the right at 197 metres. | labels ['signboard', 'signboard', 'signboard', 'signboard'] | 19.3 |
| ✅ | Where's the nearest ATM? | I haven't seen an ATM in any capture. | refused | 21.9 |
| ✅ | ಶ್ರೀ ಲಕ್ಷ್ಮಿ ಅಂಗಡಿ ಎಲ್ಲಿತ್ತು? | SRI LAKSHMI ಶ್ರೀ ಲಕ್ಷ್ಮಿ ಅಂಗಡಿ ನಿಮ್ಮ ಬಲ ಬದಿಯಲ್ಲಿ, ನಡಿಗೆಯಲ್ಲಿ ಸುಮಾರು 62 ಮೀಟರ್ ದೂರದಲ್ಲಿ 0:22 ಕ್ಕೆ ಇತ್ತು. | labels ['shop sign'] | 9.9 |
| ✅ | தேங்காய் கடை எங்கே இருந்தது? | தேங்காய் கடை உங்கள் நடையின் தொடக்கத்தில் இடது பக்கம், சுமார் 12 மீட்டரில் இருந்தது. | labels ['fruit stand'] | 10.3 |
| ✅ | Where was Kalyan Jewellers? | KALYAN was on your left, about 144 m into the walk at 2:21. | labels ['signboard'] | 10.4 |
| ✅ | Where was the Levi's store? | The Levi's OUTLET was on your left about 313 metres into the walk, around 5:06. | labels ['shop sign'] | 9.5 |
| ✅ | Which side of the road was BIBA on? | BIBA was on your left, about 187 metres into the walk. | labels ['signboard'] | 8.8 |
| ✅ | How far from my office was Louis Philippe? | Louis Philippe Outlet was on your left, about 204 m from your office on your Oct 29 walk. | labels ['signboard'] | 21.1 |
| ✅ | How do I get from Kalyan Jewellers to Levi's? | Keep walking the same way about 170 metres from Kalyan, past GRT Jewellers and BIBA, and you'll reach Levi's OUTLET on your left. | no expectation (manual check) | 28.5 |
| ✅ | Did I pass a fruit cart? | Yes, you passed a blue fruit cart on your right about 50 m into your Oct 29 walk. | labels ['fruit stall'] | 14.4 |
| ✅ | What clothing stores did I pass? | On your newest walk you passed Reebok on your left at 21m, FOREVER 21 on your left at 23m, Soch on your left at 130m, BIBA on your left at 187m, and LOUIS PHILIPPE OUTLET on your left at 204m. Further on you passed Van Heusen on your left at 246m, Manyavar on your left at 257m, Levi's OUTLET on your left at 313m, RED TAPE ONLINE OUTLET on your left at 355m and BRAND FACTORY on your left at 368m. | labels ['signboard', 'signboard', 'shop sign', 'signboard', 'signboard', 'signboard', 'signboard', 'shop sign', 'signboard', 'signboard'] | 12.5 |
| ✅ | Was there a Lenovo shop? | Yes, there was a Lenovo sign on your left about 278 m into your walk. | labels ['signboard'] | 13.0 |
| ✅ | Where's the nearest metro station? | I haven't seen a metro station in any capture. | refused | 13.1 |
| ✅ | ಕಲ್ಯಾಣ್ ಜ್ಯುವೆಲ್ಲರ್ಸ್ ಎಲ್ಲಿತ್ತು? | KALYAN ಬೋರ್ಡ್ ನಿಮ್ಮ ಎಡಭಾಗದಲ್ಲಿ, ನಡಿಗೆಯ ಸುಮಾರು 144 ಮೀಟರ್ ಒಳಗೆ, 2:21 ಕ್ಕೆ ಇತ್ತು. | labels ['signboard'] | 16.2 |
| ✅ | லீவைஸ் கடை எங்கே இருந்தது? | Levi's OUTLET கடை நடையில் சுமார் 313 மீட்டரில் உங்கள் இடது பக்கம் இருந்தது. | labels ['shop sign', 'shop front'] | 13.6 |
