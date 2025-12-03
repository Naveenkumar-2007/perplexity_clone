import re


class NameTool:
    """Extract user names from natural language messages."""

    def extract_name(self, text: str):
        """
        Extract name from sentences like:
        - i am naveen
        - I'm Naveen
        - my name is naveen
        """
        text = text.lower()

        patterns = [
            r"i am ([a-zA-Z]+)",
            r"i'm ([a-zA-Z]+)",
            r"my name is ([a-zA-Z]+)"
        ]

        for p in patterns:
            m = re.search(p, text)
            if m:
                name = m.group(1).strip().title()
                return name

        return None
