class Blog:
    title: str
    name: str
    date: str
    url: str
    # section content type list dict with key and value is string
    section_contents: list[dict]

    def __init__(self, title, name, date, url):
        self.title = title
        self.name = name
        self.date = date
        self.url = url

    def __str__(self):
        return f"Blog(url={self.url})"

    def __repr__(self):
        return str(self)

    def to_chunks(self):
        chunks: list[str] = []
        for section in self.section_contents:
            if len(section.get('content')) > 0:
                chunks += [
                    '# Blog title: ' + self.title + '\n'
                    + '## Date: ' + self.date + '\n'
                    + '## Url: ' + self.url + '\n'
                    + '## Section: ' + section.get('header') + '\n'
                    + '### Content: ' + '\n'
                    + section.get('content')
                ]

        return chunks
