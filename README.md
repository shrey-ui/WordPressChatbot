# WordPress Chatbot Integration

This project integrates a Query Suggestion Chatbot with Chain of Thought functionality into WordPress sites using a Flask backend. 
It also uses a RAG in the backend using a FAISS Vector DB. This RAG backend is combined with Chain of Thought querying using RAT (Retrieval Augmented Thoughts).

![alt text](https://github.com/shrey-ui/WordPressChatbot/assets/RAT_pic.png?raw=true)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Components](#components)

## Introduction

The WordPress Chatbot Integration project aims to provide a seamless way to add an intelligent chatbot to WordPress sites. The chatbot leverages a Retrieval-Augmented Generation (RAG) model and Chain of Thought reasoning to suggest relevant queries and responses based on the content of the WordPress site.

Please NOTE - This is not 

## Features

- Dynamic Query Suggestions: The chatbot suggests relevant queries based on user input. Uses Retrieval Augmented Generation for this.
- Chain of Thought Reasoning: The chatbot maintains context and logical progression in conversations.
- Real-time Content Updates: The chatbot fetches real-time content from the WordPress site.
- Customizable: Easily configurable to adapt to different WordPress sites. In fact by passing your WordPress site the first time, 
				embeddings will be generated automatically.

## Installation

### Prerequisites

- WordPress site
- Python 3.x
- Flask
- Required Python packages (listed in `requirements.txt`)

### Steps

1. Clone the Repository

    ```bash
<<<<<<< HEAD
    git clone https://github.com/shrey-ui/WordPressChatbot.git
    cd WordPressChatbot
=======
    git clone https://github.com/shrey-ui/wasserstoff-ai-task.git
    cd wasserstoff-ai-task
>>>>>>> 677a9e0 (updated commit)
    ```

2. Install Python Packages

    ```bash
    pip install -r requirements.txt
    ```

3. Setup Flask Backend

    Ensure the Flask app is running on a server accessible from your WordPress site.

    ```bash
    python app.py
    ```
<<<<<<< HEAD
4. WordPress Site included - 
    The WordPress site used for development purpose is included in the Repository
    at wordpress/wordpress-site. The plugins are given seperately but they are in the 
    wordpress-site location as well.

    You are advised to use something like Local to get the WordPress site running at Port 10003
    for testing of the App itself.


5. WordPress Plugin Installation
=======

4. WordPress Plugin Installation
>>>>>>> 677a9e0 (updated commit)

    - Upload the `chatbot-plugin` directory to the `/wp-content/plugins/` directory.
    - Activate the plugin through the 'Plugins' menu in WordPress.

<<<<<<< HEAD
6. Add Chatbot to WordPress
=======
5. Add Chatbot to WordPress
>>>>>>> 677a9e0 (updated commit)

    The plugin automatically adds the chatbot to the footer of your WordPress site. Ensure your theme supports the `wp_footer` action.

## Configuration

### Flask Configuration

Modify the Flask endpoint URL in the `chatbot.js` file:

```javascript
fetch('http://your-flask-api-endpoint', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestData),
})
```

### WordPress Configuration

No additional configuration is required for the WordPress plugin. It automatically adds the chatbot to the footer.

## Usage

1. Interacting with the Chatbot

    - Navigate to your WordPress site.
    - The chatbot should appear in the bottom right corner.
    - Type a message in the input field and click "Send".

2. Updating the Vector Database

    New posts are automatically added to the vector database when published. 

## Components

### Flask Backend

<<<<<<< HEAD
- `chatbot_flask_api.py.py`: Main Flask application file.
=======
- `app.py`: Main Flask application file.
>>>>>>> 677a9e0 (updated commit)
- `gen_vector_db.py`: Module for generating and managing the vector database.
- `RAG_gen.py`: Module for handling RAG and Chain of Thought reasoning.

### WordPress Plugin
	
	Chatbot Plugin - in wordpress/chatbot-plugin

	- `chatbot-plugin.php`: Main plugin file for WordPress.
	- `chatbot.js`: JavaScript file for handling chatbot interactions.

	Update Embedding on Post Publish - in wordpress/vector-db-updater




