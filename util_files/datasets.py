from csv import DictReader


class datasets():
    def __init__(self, path="./data"):
        self.path = path

        print("start reading dataset")
        bodies_file = "train_bodies.csv"
        stances_file = "train_stances.csv"

        self.stances = self.read(stances_file)
        articles = self.read(bodies_file)
        # make an arra of all articles
        self.articles = dict()

        for stance in self.stances:
            stance['Body ID'] = int(stance['Body ID'])
#makes body id aninteger value
        for article in articles:
            self.articles[int(article["Body ID"])] = article['articleBody']

        print("Total stances" + str(len(self.stances)))
        print("Total Articles:" + str(len(self.articles)))

    def read(self, filename):
        rows_array = []
        with open(self.path + "/" + filename, "r", encoding='utf-8') as  table:
            reader = DictReader(table)

            for line in reader:
                rows_array.append(line)

        return rows_array
