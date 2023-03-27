
import re
from flask import Flask, abort, redirect, render_template, request, current_app
from html import escape
from werkzeug.exceptions import default_exceptions, HTTPException
from pypinyin import pinyin, Style

from config import confusion_dic_path
from lm_corrector import LMCorrector
from macbert_corrector import MacBertCorrector
from utils import find_difference, substrings

app = Flask(__name__)


app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


@app.route('/favicon.ico')
def favicon():
    # 后端返回文件给前端（浏览器），send_static_file是Flask框架自带的函数
    return current_app.send_static_file('static/favicon.ico')


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/correcting", methods=["POST"])
def compare():
    """进行纠错"""
    # 读取文件
    if request.files["file1"]:
        try:
            file1 = request.files["file1"].read().decode("utf-8")
        except Exception:
            abort(400, "invalid file")
    elif request.form.get("text1"):
        file1 = request.form.get("text1")
    else:
        abort(400, "missing file")

    if request.form.get("algorithm") == "lm":
        corrector = LMCorrector()
    elif request.form.get("algorithm") == "macbert":
        corrector = MacBertCorrector()
    elif request.form.get("algorithm") == "lm_macbert":
        corrector = LMCorrector()
    else:
        abort(400, "invalid algorithm")

    sentence_lst = file1.strip().split('\n')
    # print(sentence_lst)
    corrected_lst = []
    highlights1 = ''    # 高亮显示错误
    for sentence in sentence_lst:
        print('\n原句: ' + sentence)
        corrected, errs = corrector.correct(sentence)    # 得到改正后的句子corrected，错误errs
        
        if errs == []:
            print('正确')
            corrected_lst.append(sentence)
            highlights1 += sentence
        else:
            print('改正：' + str(corrected) + '\n错误：' + str(errs))
            corrected_lst.append(corrected)
            highlights1 += escape(sentence[0 : errs[0][2]])
            for i in range(0,len(errs)):
                # 开始位置errs[i][2]，结束位置errs[i][3]
                highlights1 += f"<span>{escape(sentence[errs[i][2] : errs[i][3]])}</span>"
                if i < len(errs)-1:
                    highlights1 += escape(sentence[errs[i][3] : errs[i+1][2]])
                else:
                    highlights1 += escape(sentence[errs[i][3] : len(sentence)])
            highlights1 += '\n'

    file2 = '\n'.join(corrected_lst)
    highlights2 = file2

    return render_template("index.html", file1=highlights1, file2=highlights2)


def highlight(s, regexes):
    """高亮显示"""

    # Get intervals for which strings match
    intervals = []
    for regex in regexes:
        if not regex:
            continue
        matches = re.finditer(regex, s, re.MULTILINE)
        for match in matches:
            intervals.append((match.start(), match.end()))
    intervals.sort(key=lambda x: x[0])

    # Combine intervals to get highlighted areas
    highlights = []
    for interval in intervals:
        if not highlights:
            highlights.append(interval)
            continue
        last = highlights[-1]

        # If intervals overlap, then merge them
        if interval[0] <= last[1]:
            new_interval = (last[0], interval[1])
            highlights[-1] = new_interval

        # Else, start a new highlight
        else:
            highlights.append(interval)

    # Maintain list of regions: each is a start index, end index, highlight
    regions = []

    # If no highlights at all, then keep nothing highlighted
    if not highlights:
        regions = [(0, len(s), False)]

    # If first region is not highlighted, designate it as such
    elif highlights[0][0] != 0:
        regions = [(0, highlights[0][0], False)]

    # Loop through all highlights and add regions
    for start, end in highlights:
        if start != 0:
            prev_end = regions[-1][1]
            if start != prev_end:
                regions.append((prev_end, start, False))
        regions.append((start, end, True))

    # Add final unhighlighted region if necessary
    if regions[-1][1] != len(s):
        regions.append((regions[-1][1], len(s), False))

    # Combine regions into final result
    result = ""
    for start, end, highlighted in regions:
        escaped = escape(s[start:end])
        if highlighted:
            result += escaped
            # result += f"<span>{escaped}</span>"
        else:
            result += f"<span>{escaped}</span>"
            # result += escaped
    return result


@app.route("/setting", methods=["GET"])
def setting():
    """设置"""
    confusion_lst = []
    for confusion in open(confusion_dic_path, "r", encoding='utf-8'):
        confusion_lst.append(confusion.split())

    del confusion_lst[0]
    confusion_lst.sort(key=lambda keys:[pinyin(i, style=Style.TONE3) for i in keys])
    return render_template("setting.html", confusion_lst=confusion_lst, i=1)


@app.errorhandler(HTTPException)
def errorhandler(error):
    """Handle errors"""
    return render_template("error.html", error=error), error.code


# https://github.com/pallets/flask/pull/2314
for code in default_exceptions:
    app.errorhandler(code)(errorhandler)

if __name__ == '__main__':
    app.run(debug=True)