# coding=utf-8

import requests
import time


def main():
    text = 'it is so bad.'
    # text = 'nice buying experience.'

    input_data = {"text": text}

    url = "http://localhost:9898/text_classification"
    headers = {'Content-Type': 'application/json;charset=UTF-8'}

    time_start = time.time()

    # send post request
    res = requests.request("post", url, headers=headers, json=input_data)

    time_end = time.time()

    print('input_text:   {}'.format(text))
    print('status_code:  {}'.format(res.status_code))
    print('response:     {}'.format(res.json()))
    print('prediction:   {}'.format(res.json()['prediction']))
    print('totally cost: {}ms'.format(int((time_end - time_start) * 1000)))


if __name__ == '__main__':
    main()
