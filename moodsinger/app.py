from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template("index.html")





@app.route('/upcoming_events')
def upcoming_events():
    return render_template("event.html")

@app.route('/new_release')
def new_release():
    return render_template("new_release.html")


@app.route('/geo_listen')
def geo_listen():
    return render_template("geo_listen.html")


@app.route('/subscribe')
def subscribe():
    return render_template("subscribe.html")


@app.route('/categories')
def categories():
    return render_template("categories.html")


@app.route('/library')
def load_library():
    return render_template("library.html")

@app.route('/notifications')
def notifications():
    return render_template("event.html")

@app.route('/profile')
def profile():
    return render_template("event.html")

@app.route('/log_out')
def log_out():
    return render_template("event.html")


if __name__ == '__main__':
    app.run()
