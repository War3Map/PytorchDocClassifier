import math
import os.path


from additional.settings import (
    DIVIDE_TEXT_DOCUMENT,
    DIVIDE_PARTS_PATH,
)


def write_lines(filename, data: list):
    """
    Write lines to file
    :param filename: filepath
    :param data: lines
    :return: None
    """
    with open(filename, "a", encoding="utf-8") as data_file:
        data_file.writelines(data)


def divide_line(line: str, line_size: int,
                strategy: str = "simple"):
    """
    Splits line by parts of less or equal line_size length

    :param line: source line
    :param line_size: size of new line part
    :param strategy: divide strategy: simple - by symbols
    smart -  words

    :return: list of str
    """
    split_func = {
        "simple": simple_divide_line,
        "smart": smart_divide_line
    }
    chosen_strat = strategy if strategy in split_func else "simple"
    return split_func[chosen_strat](line, line_size)


def simple_divide_line(line: str, line_size: int):
    """
    Splits line by parts of line_size length
    :return: list of str
    """
    new_lines = []
    parts_count = math.ceil(len(line) / line_size)
    for part_idx in range(parts_count):
        new_lines.append(
            line[part_idx * line_size: (part_idx + 1) * line_size]
        )
    return new_lines


def smart_divide_line(line: str, line_size: int):
    """
    Splits line by parts of less than line_size length with
    complete words

    :return: list of str
    """
    new_lines = []
    words = line.split()
    part_len = 0
    part = []
    for word in words:
        # add 1 to count deleted space between words
        part_len += len(word) + 1

        if part_len <= line_size:
            part.append(word)
        else:
            part.append("\n")
            new_lines.append(" ".join(part))
            part = [word]
            part_len = len(word) + 1

    # if part not empty add last words
    if part:
        part.append("\n")
        new_lines.append(" ".join(part))

    return new_lines


def split_text_file(filename,
                    folder,
                    part_size=50,
                    line_size=None,
                    doc_name="doc",
                    encoding="utf-8"):
    """

    :param filename:
    :param folder:
    :param part_size: length of part in lines
    :param doc_name:
    :param encoding:
    :return:
    """
    prefix = os.path.join(folder, doc_name)
    new_filename = "{prefix}_{id}.txt"

    with open(filename, "r", encoding=encoding) as data_file:
        lines = []
        part_count = 0
        prevline = ""
        line_counter = 0
        for line in data_file:

            # delete empty lines if it doest go twice
            if line == "\n" and prevline != "\n":
                prevline = line
                continue

            # divide big line if needed
            if line_size is not None:
                new_lines = (divide_line(line, line_size, strategy="smart")
                             if len(line) > line_size
                             else [line])
            else:
                new_lines = [line]

            # write lines to file
            for corrected_line in new_lines:
                line_counter += 1
                lines.append(corrected_line)
                if line_counter % part_size == 0:
                    part_count += 1
                    part_filename = new_filename.format(prefix=prefix,
                                                        id=part_count)
                    write_lines(part_filename, lines)
                    lines = []
            prevline = line

        if lines:
            write_lines(new_filename.format(prefix=prefix, id=part_count + 1),
                        lines)


def text_to_image(directory):
    """
    Convert all text files in directory to images
    :return:
    """
    ...


if __name__=='__main__':
    split_text_file(DIVIDE_TEXT_DOCUMENT,
                    DIVIDE_PARTS_PATH,
                    part_size=30,
                    line_size=50,
                    encoding="CP1251",
                    doc_name="source1", )
