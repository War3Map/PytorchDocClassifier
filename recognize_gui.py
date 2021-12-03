import queue
import threading
from pathlib import Path

import PySimpleGUI as sg
import subprocess

from recognizer.recognize_printed import recognize_image


processing = False
sg.theme('LightBlue1')  # Add a touch of color
# All the stuff inside your window.
layout = [
    [sg.Text('Распознать')],
    [sg.Text('Путь к файлу'), sg.InputText(key='path_to_file'),
     sg.FilesBrowse("Выбрать", file_types=(("Изображения",
                                           "*.jpeg;*.jpg;*.png;*.gif;*.tiff"), ))],

    [sg.Text('Результат(имя)'), sg.InputText(key='path_to_result')],

    [sg.Text('Каталог для сохранения'), sg.InputText(key='save_directory'),
     sg.FolderBrowse("Выбрать")],
    [sg.ProgressBar(100, orientation='h', size=(50, 20), key='progressbar')],

    [sg.Button('Распознать'), sg.Button('Выйти')]
]

# Create the Window
window = sg.Window('Конвертер', layout)
# result values
out_queue = None
# Thread for search
searching_thread = None


def process_recognition(files_list: list,
                        result_name,
                        result_dir,
                        window_ref,
                        progress_bar,
                        res_queue):
    """
    Recognize images from file list
    :param res_queue: queue for write results
    :param window_ref: window object
    :param result_dir: directory to save results
    :param result_name: common name for results
    :param files_list: list of file paths
    :return: None
    """
    results = []
    max_progress = len(files_list)
    for idx, filename in enumerate(files_list):
        new_filename = f"result{idx}.txt"
        recognize_image(filename, new_filename)
        results.append(new_filename)
        current_progress = (idx + 1) / max_progress * 100
        # print(current_progress)
        progress_bar.UpdateBar(current_progress)

    # search_core.SomeClass.SomeFunction()
    res_queue.put(results)
    # put a message into queue for GUI
    window_ref.write_event_value('thread_run', 'done')


# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Выйти':  # if user closes window or clicks cancel
        break
    # распознавание
    if event == 'Распознать' and processing:
        sg.Popup('Идёт обработка')
    elif event == 'Распознать' and not processing:
        if 'path_to_file' in values:
            processing = True
            files_to_recognize = [
                str(Path(path).resolve())
                for path in values['path_to_file'].split(';')]
            # print(files_to_recognize)

            new_name = values['path_to_result']
            out_dir = values['save_directory']

            # Queue used for getting results
            out_queue = queue.Queue()
            progres_bar = window['progressbar']
            searching_thread = threading.Thread(target=process_recognition,
                                                args=(files_to_recognize,
                                                      new_name,
                                                      out_dir,
                                                      window,
                                                      progres_bar,
                                                      out_queue),
                                                daemon=True)
            searching_thread.start()

    if event == "thread_run":
        result_list = out_queue.get()
        # window['-LIST-'].update(result_list)
        sg.Popup('Распознавание завершено!')
        searching_thread = None

window.close()
