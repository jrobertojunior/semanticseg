from xml.dom.minidom import parse

def parse_training_data(path):
    """Parses the training xml and returns a dictionary of the data

    The dict has the following format:
    {
        img: [list of image paths],
        mask: [list of mask paths]
    }
    """
    document = parse(path)

    images = document.getElementsByTagName("srcimg")
    labels = document.getElementsByTagName("labelimg")

    training_data = {"img": [], "mask": []}

    training_data["img"] = [images[i].getAttribute("name") for i in range(len(images))]
    training_data["mask"] = [labels[i].getAttribute("name") for i in range(len(labels))]

    return training_data

# xml for testing
# training_data = get_training_data("data/training2.xml")
# print(training_data)