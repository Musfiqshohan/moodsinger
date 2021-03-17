from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/library')
def load_library():
    return render_template("library.html")



@app.route('/upcoming_events')
def upcoming_events():
    return render_template("event.html")


if __name__ == '__main__':
    app.run()
