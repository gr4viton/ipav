from chain.thread_safe_data import LockedValue


class Chain:
    tag_names = []
    chain_names = []
    load_data_chain_names = []

    data = LockedValue()

    delimiter = ","
    strip_chars = " \t"

    def load_steps_from_file(self, path):
        with open(path, "r") as myfile:
            string = myfile.read
        self.load_steps_from_string(string)

    def load_steps_from_string(self, string):
        self.step_names_list = string.replace("\n", self.delimiter).split(
            self.delimiter
        )
        # print(self.step_names)
        self.step_names_list = [
            step_name.strip(self.strip_chars) for step_name in self.step_names_list
        ]

        self.step_names = [step for step in self.step_names_list if step != ""]
        # print(self.step_names)

    def __init__(
        self,
        name,
        start_chain=True,
        path="",
    ):
        self.name = name
        self.tag_search = name in self.tag_names

        if path != "":
            self.load_steps_from_file(path)
        else:
            if self.name in ["standard"]:

                # self.step_names = ['original', 'resize', 'gray', 'thresholded', 'sobel vertical']
                # self.step_names = ['original', 'resize', 'gray', 'thresholded', 'sobel horizontal']\
                self.step_names = [
                    "original",
                    "resize",
                    "gray",
                    "thresholded",
                    "laplacian",
                ]
                self.step_names = ["original", "gauss", "resize"]

                # self.step_names = ['original', 'resize', 'gray', 'detect red', 'blender cube']
                # self.step_names = ['original', 'resize', 'gray', 'thresholded', 'blender cube']
                # self.step_names = ['original', 'resize', 'detect red']
                # self.step_names = ['original', 'resize', 'rgb stack']

                # string = 'original, resize, gauss, resize'
                string = "original"
                if start_chain:
                    self.load_steps_from_string(string)

        if self.name in self.load_data_chain_names:
            self.load_data()

    def load_data(self):
        # self.data = LockedValue( [1,2,3,42,69] )
        self.data = [1, 2, 3, 42, 69]
        pass
