class Message:
    def __init__(self, text: str, sender: str) -> None:
        self.text = text
        self.sender = sender

    def to_dict(self) -> dict:
        return {"text": self.text, "sender": self.sender}
