import flask
import pandas as pd
app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return (flask.render_template('main.html'))

    if flask.request.method == 'POST':
        blog_url = flask.request.form['url']

        input_variable = pd.DataFrame([blog_url])

        prediction = model.predict(input_variable)[0]

        return flask.render_template('main.html', input_blog_url = { 'input_blog_url': blog_url})
