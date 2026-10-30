from loci.agent.agent import language_problems, question_language


def test_question_language() -> None:
    assert question_language("அடுப்பு எரிந்ததா?") == "Tamil"
    assert question_language("ಸ್ಟವ್ ಆನ್ ಆಗಿತ್ತಾ?") == "Kannada"
    assert question_language("Is the stove on?") is None


def test_answer_must_be_in_the_language_of_the_question() -> None:
    q = "லீவைஸ் கடை எங்கே இருந்தது?"
    assert language_problems(q, "ನಾನು ಯಾವುದೇ ಕ್ಯಾಪ್ಚರ್‌ನಲ್ಲಿ ನೋಡಲಿಲ್ಲ")  # Kannada answer to a Tamil question
    assert language_problems(q, "I have not seen it")
    assert language_problems(q, "நான் பார்க்கவில்லை") == []
    assert language_problems(q, "Kalyan கடை இடதுபுறம் இருந்தது") == []
    assert language_problems("Where was it?", "On your left") == []
