# Eval results (questions.yaml)

- questions: 35
- accuracy: 80%
- refusals correct: 4/4
- verified on first attempt: 83%
- median latency: 22.3 s
- cloud calls: 0 (local only)

| | question | answer | why | s |
| --- | --- | --- | --- | --- |
| ✅ | What's on my desk? | On the hallway desk I last saw at 0:18, there are plastic bags on the desk, and a wall socket is also on the desk. | labels ['desk', 'plastic bag', 'plastic bag', 'wall socket'] | 26.4 |
| ✅ | Where is my laptop? | Last seen left of the desk in the bedroom at 0:23 in the home2 walk on 2026-10-29. The laptop is movable, so it may have moved since then. | labels ['laptop', 'laptop'] | 16.1 |
| ✅ | Where's the keyboard? | The most recent keyboard I have is left of the desk in the bedroom, seen at 0:24 in the home2 walk. | labels ['keyboard', 'keyboard', 'keyboard', 'keyboard', 'keyboard'] | 12.0 |
| ✅ | Is the laundry done? | I haven't seen any indication the laundry is done; the last sighting I have is a laundry pile still in the bedroom at 0:21, and there's no record of a washing machine or a note about laundry being finished. | labels ['laundry pile', 'laundry bucket', 'clothes pile'] | 27.4 |
| ❌ | Where's the purple bucket? | I haven't seen a purple bucket in any capture. | refused | 35.4 |
| ✅ | Where did I leave my backpack? | The most recent sighting is in the hallway left of the desk. It was last seen at 0:18 in the home capture on 2026-10-28T16:45:46+05:30, at left of the desk, in the hallway. It’s movable, so check – that was the last time it was seen. | labels ['backpack', 'backpack', 'backpack', 'backpack'] | 16.4 |
| ✅ | Where's the laptop stand? | The laptop stand was last seen on the desk in the office, at 0:06 in the 2026-10-28 home capture. It’s movable and that sighting is old, so it may have moved since then. | labels ['laptop stand'] | 14.4 |
| ✅ | Is the stove on? | As of the most recent kitchen capture (home2, 2026-10-29, seen at 0:18), the gas stove in the kitchen is off – burner_lit is false and nothing is on a burner. It was on in the previous capture (home, 2026-10-28 at 0:15 with the right burner lit). | no expectation (manual check) | 14.2 |
| ❌ | Are the lights on? | I haven't seen a light in any capture. | refused | 8.2 |
| ✅ | How far is the desk from the door? | In the hallway capture the desk is about 1.3 m from the nearby door — the straight-line distance is ~1.3 m, listed as ~1.34 m in the object record, as seen at 0:18 in the hallway. | no expectation (manual check) | 30.2 |
| ✅ | Where's my passport? | I haven't seen a passport in any capture. | refused | 6.7 |
| ✅ | Where are my car keys? | I haven't seen a car keys in any capture. | refused | 10.5 |
| ✅ | ಸ್ಟವ್ ಆನ್ ಆಗಿತ್ತಾ? | ಮನೆಯ ಹೊಸ capture (2026-10-29) ನಲ್ಲಿ ಸ್ಟವ್ ನೋಡಿದ ಸಮಯ 0:18 ರಂತೆ ಬರ್ನರ್‌ಗಳು ಲಿಟ್ ಆಗಿರಲಿಲ್ಲ — ಸ್ಟವ್ ಆಫ್ ಆಗಿತ್ತು. ಹಿಂದಿನ capture ನಲ್ಲಿ 0:15 ರಂದು ಬಲ ಬರ್ನರ್ ಲಿಟ್ ಆಗಿತ್ತು. | labels ['gas stove', 'gas stove', 'gas stove', 'gas stove'] | 16.8 |
| ✅ | அடுப்பு எரிந்துகொண்டிருந்ததா? | சமீபத்திய கைப்பற்றல் home2 இல் அடுப்பு 0:18 இல் எரியவில்லை — burner_lit false. அது சமையலறையில் உள்ளது, அந்த நேரத்தில் எரிந்துகொண்டிருக்கவில்லை. | labels ['gas stove', 'gas stove'] | 53.0 |
| ✅ | என் பை எங்கே? | உங்கள் பையை கடைசியாக ஹால்வேயில் டெஸ்கின் இடது பக்கம் கண்டேன்; அது 2026-10-28 கேப்ச்சரில் 0:18-ல் பார்க்கப்பட்டது. அது நகரக்கூடியது, அதனால் இப்போது அங்கே இருக்காது. | labels ['backpack', 'backpack', 'backpack', 'backpack'] | 26.3 |
| ❌ | What changed in my home since the first capture? | I'm not sure. The last confirmed sighting of the gas stove was in the kitchen, at 0:18. | not sure (unverified) | 52.2 |
| ❌ | Where was the coconut stall? | I haven't seen a coconut stall in any capture. | refused | 42.0 |
| ✅ | What shops did I pass? | On your newest street walk, started from my office (Novel MSR Park) on 2026-10-29, you passed shops in order along the left/right side. Early examples: BHIMA on the left at 74.4 m, 1:12; BLUESTONE on the left at 77.8 m, 1:18; megemart / MegaMart on the left at 86.6 m, 1:25; LAVAS on the left at 97.8 m, 1:35; KALYAN on the left at 111.0 m, 1:47; Dr Sreesha on the left at 129.9 m, 2:09; FS FAVOURITE SHOP ಫೇವರೈಟ್ ಶಾಪ್ on the left at 129.9 m, 2:09; GRT JEWELLARS on the left at 143.5 m, 2:21 and so on through Louis Philippe OUTLET on the right at 204.2 m, 3:25, Manyavar on the left at 257.3 m, 4:24, Lenovo on the left at 274.2 m, 4:38, RED TAPE ONLINE OUTLET on the left at 355.0 m, 5:47 and BRAND FACTORY on the left at 368.0 m, 6:07. | labels ['signboard', 'signboard', 'shop sign', 'signboard', 'signboard', 'signboard', 'signboard', 'shop sign', 'signboard', 'signboard', 'shop sign', 'signboard', 'shop sign', 'signboard'] | 22.3 |
| ❌ | Was the coconut stall open? | I haven't seen a coconut stall in any capture. | refused | 45.5 |
| ❌ | How do I get from the coconut stall to the next shop? | I haven't seen a coconut stall in any capture. | refused | 40.6 |
| ✅ | Which side of the road were the shops on? | The walk captured on 2026-10-28 had shops on both sides, with a cluster on the right early on and some on the left later, while the 2026-10-29 walk from your office shows shops mainly on the left. | labels ['shop sign', 'shop sign', 'shop front'] | 16.3 |
| ✅ | Where's the nearest ATM? | I haven't seen an ATM in any capture. | refused | 7.9 |
| ✅ | ಶ್ರೀ ಲಕ್ಷ್ಮಿ ಅಂಗಡಿ ಎಲ್ಲಿತ್ತು? | SRI LAKSHMI ಶ್ರೀ ಲಕ್ಷ್ಮಿ ಎಂಬ ಬೋರ್ಡನ್ನು ನೀವು street ವಾಕ್‌ನಲ್ಲಿ ಬಲಗಡೆ, ವಾಕ್‌ನಲ್ಲಿ ಸುಮಾರು 62 ಮೀಟರ್ ಒಳಗೆ ನೋಡಿದ್ದೀರಿ; ಅದು 0:22ರ ಸಮಯದಲ್ಲಿ ಕಂಡುಬಂದಿತು. | labels ['signboard'] | 25.7 |
| ❌ | தேங்காய் கடை எங்கே இருந்தது? | என் கேப்ச்சர்களில் தேங்காய் கடை எதுவும் காணப்படவில்லை. | refused | 46.8 |
| ✅ | Where was Kalyan Jewellers? | Kalyan Jewellers was on your left about 151 m into the street walk, seen at 2:27. | labels ['signboard', 'signboard', 'shop sign', 'signboard', 'shop sign', 'signboard'] | 19.5 |
| ✅ | Where was the Levi's store? | The Levi's sign was on your left about 314 m into the street2 walk, last seen around 5:03. | labels ['signboard', 'shop sign', 'signboard'] | 10.2 |
| ✅ | Which side of the road was BIBA on? | BIBA was on your left, about 187 m into the walk (seen at 3:01). | labels ['shop sign', 'signboard', 'shop sign'] | 21.0 |
| ✅ | How far from my office was Louis Philippe? | Louis Philippe came up about 173 m from your office (Novel MSR Park) on your left, about 2:49 into the walk; street2:shop_sign_91 | labels ['shop sign', 'signboard', 'signboard'] | 12.3 |
| ✅ | How do I get from Kalyan Jewellers to Levi's? | From the Kalyan sign on your left at about 151 m into the walk, keep going the same way you walked. Levi's is further ahead on the same left side, about 314 m into the walk — roughly 163 m ahead of Kalyan. You’ll pass a long stretch of shops on the left including GRT Jewellers and BIBA before reaching Levi's on the left. | no expectation (manual check) | 29.2 |
| ✅ | Did I pass a fruit cart? | Yes. You passed a fruit cart on your left about 35 m into the street2 walk, seen at 0:32. | labels ['fruit stall', 'fruit stall', 'fruit stand', 'fruit stall', 'fruit stall'] | 27.6 |
| ✅ | What clothing stores did I pass? | On the 2026-10-29 walk starting from your office (Novel MSR Park) you passed the following clothing stores in walk order: BIBA on the left at 173.2 m into the walk, passed at 2:49; LOUIS PHILIPPE OUTLET on the right at 204.2 m, passed at 3:25; Manyavar on the left at 257.3 m, passed at 4:24; BRAND FACTORY on the left at 368.0 m, passed at 6:07. | labels ['signboard', 'signboard', 'signboard', 'shop sign', 'signboard'] | 21.1 |
| ✅ | Was there a Lenovo shop? | Yes — on the street walk you passed a Lenovo storefront on your left about 274 m into the walk, seen at 4:45. | labels ['signboard', 'shop sign', 'shop sign', 'signboard'] | 47.1 |
| ✅ | Where's the nearest metro station? | I haven't seen a metro station in any capture. | refused | 9.2 |
| ✅ | ಕಲ್ಯಾಣ್ ಜ್ಯುವೆಲ್ಲರ್ಸ್ ಎಲ್ಲಿತ್ತು? | street2 ನಡಿಗೆಯಲ್ಲಿ ನಿಮ್ಮ ಎಡಕ್ಕೆ ಸುಮಾರು 144 ಮೀ ನಡೆದ ನಂತರ 'KALYAN ಕಲ್ಯಾಣ' ಎಂಬ ಹೆಸರಿನ ಕಲ್ಯಾಣ್ ಜ್ಯುವೆಲ್ಲರ್ಸ್‌ನ ಸೈನ್ ಕಾಣುತ್ತದೆ; 2:21ಕ್ಕೆ ಕಂಡಿದ್ದು ಎಡಭಾಗದಲ್ಲಿದೆ. | labels ['signboard', 'signboard', 'signboard', 'shop sign', 'signboard'] | 49.5 |
| ✅ | லீவைஸ் கடை எங்கே இருந்தது? | Levi's கடையின் பலகை street2 நடையில் இடது பக்கம், நடை தொடங்கிய இடத்திலிருந்து சுமார் 314 மீட்டர் உள்ளே, 5:03 இல் காணப்பட்டது. | labels ['signboard', 'shop sign', 'signboard', 'shop sign'] | 27.5 |
