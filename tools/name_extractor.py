import re

class NameExtractor:
    def extract(self, text: str):
        # Format: "i am naveen" , "my name is naveen"
        match = re.search(r"(i am|my name is)\s+([A-Za-z]+)", text.lower())
        if match:
            return match.group(2).title()
        return None
