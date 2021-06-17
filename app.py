# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os

app = Flask(__name__)
# Eliminate CORS issue.
# CORS(app)
@app.route('/', methods=['GET'])
def start():
    return "Hello World!"
    # return jsonify({
    #     "Message": f"Welcome to our awesome platform!!",
    #     # Add this option to distinct the POST request
    #     "METHOD" : "POST"
    # })