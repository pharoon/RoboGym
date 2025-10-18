# RoboGym 🤖

[![Stars](https://img.shields.io/github/stars/pharoon/RoboGym?style=social)](https://github.com/pharoon/RoboGym/stargazers)
[![Forks](https://img.shields.io/github/forks/pharoon/RoboGym?style=social)](https://github.com/pharoon/RoboGym/network/members)

## Description 📝

RoboGym is a desktop simulation tool for robotic arms, designed to facilitate the training and testing of Reinforcement Learning (RL) models in realistic environments. It offers real-time feedback, supports multiple simulations, and seamlessly integrates AI models with robotic arm software. This platform provides a scalable and efficient solution for advancing robotics in research, industry, and academia.

## Table of Contents 📚

1.  [Description](#description-)
2.  [Features](#features-)
3.  [Tech Stack](#tech-stack-)
4.  [Installation](#installation-)
5.  [Usage](#usage-)
6.  [Project Structure](#project-structure-)
7.  [API Reference](#api-reference-)
8.  [Contributing](#contributing-)
9.  [License](#license-)
10. [Important Links](#important-links-)
11. [Footer](#footer-)

## Features ✨

*   **Realistic Robotic Arm Simulation:** Provides a desktop-based simulation environment for robotic arms.
*   **Reinforcement Learning Support:** Enables training, testing, and optimization of RL models.
*   **Real-Time Feedback:** Offers real-time feedback during simulations.
*   **Multiple Simulations:** Supports running multiple simulations simultaneously.
*   **AI Model Integration:** Facilitates seamless integration of AI models with robotic arm software.
*   **Scalability:** Designed for scalability to accommodate complex robotic systems.
*   **Cross-Platform Compatibility:** Builds for Windows, macOS and Linux
*   **Model Management:** Ability to upload, download, rename and delete trained models
*   **Training Visualization:** Provides visual charts of training progress
*   **User Authentication:** Implements user login and registration functionality

## Tech Stack 💻

*   **Primary Language:** Python
*   **Frontend:** TypeScript, React, React-Bootstrap, Bootstrap
*   **Backend:** Python, Flask
*   **Desktop Framework:** Electron
*   **Build Tool:** Electron-Vite
*   **UI Libraries:** Material UI, Emotion
*   **Charting:** Recharts

## Installation ⚙️

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/pharoon/RoboGym.git
    cd RoboGym
    ```

2.  **Install dependencies:**

    ```bash
    npm install
    ```

## Usage 🚀

1.  **Development:**

    ```bash
    npm run dev
    ```

    This command starts the application in development mode with hot reloading.

2.  **Building for Production:**

    *   **For Windows:**

        ```bash
        npm run build:win
        ```

    *   **For macOS:**

        ```bash
        npm run build:mac
        ```

    *   **For Linux:**

        ```bash
        npm run build:linux
        ```

    These commands build the application for the specified platform.

3. **Running RoboGym**

   The RoboGym application allows users to train, test, and analyze reinforcement learning models for robotic arm control. After building the application, execute it and follow the UI for uploading models, initiating training sessions and running test scenarios. The application connects to a local Python backend to run simulations and manage the models.
   - **Login/Register**: Start by creating an account or logging in.
   - **Dashboard**: Access the dashboard to view active models, training sessions, and test run statistics.
   - **Train Model**: Upload a model or start training a new one with custom configurations.
   - **Test Model**: Evaluate trained models in a simulated environment.
   - **Analytics**: Analyze training performance with reward charts.
   - **All Models**: Manage existing trained models

## Project Structure 📂

```
RoboGym/
├── .editorconfig
├── .eslintignore
├── .prettierignore
├── .prettierrc.yaml
├── electron.vite.config.ts
├── eslint.config.mjs
├── package.json
├── README.md
├── src/
│   ├── main/
│   │   ├── main.ts
│   │   └── Utils/
│   │       └── Websocket.ts
│   ├── preload/
│   │   ├── index.d.ts
│   │   └── index.ts
│   └── renderer/
│       ├── index.html
│       └── src/
│           ├── App.tsx
│           ├── app.css
│           ├── assets/
│           │   ├── base.css
│           │   └── main.css
│           ├── components/
│           │   ├── AnalyticsPage/
│           │   │   ├── Analytics.css
│           │   │   └── Analytics.tsx
│           │   ├── HomePage/
│           │   │   ├── HomePage.css
│           │   │   └── HomePage.tsx
│           │   ├── Layout/
│           │   │   ├── MainLayout.css
│           │   │   └── MainLayout.tsx
│           │   ├── ListModelsPage/
│           │   │   ├── AllModel.css
│           │   │   ├── AllModel.tsx
│           │   │   ├── EmptyState.tsx
│           │   │   └── ModelCard.tsx
│           │   ├── LoginPage/
│           │   │   ├── LoginPage.css
│           │   │   └── LoginPage.tsx
│           │   ├── Modals/
│           │   │   ├── DeleteModel.css
│           │   │   ├── DeleteModel.tsx
│           │   │   ├── Modal.css
│           │   │   ├── Modal.tsx
│           │   │   └── NameInputModal.tsx
│           │   └── Trainpage/
│           │       ├── Train.css
│           │       └── Train.tsx
│           ├── env.d.ts
│           ├── main.tsx
│           └── utils/
│               ├── FetchData.ts
│               ├── interfaces.ts
│               └── LoadingScreen.tsx
├── tsconfig.json
├── tsconfig.node.json
└── tsconfig.web.json
```

## API Reference ⚙️

The RoboGym project includes a Flask API that provides endpoints for user authentication, model management, and training/testing. The API is available at `http://localhost:5000`.

### Endpoints:

*   `/register` (POST): Registers a new user.
*   `/login` (POST): Logs in an existing user and returns a token.
*   `/train` (GET): Starts a new training session with specified parameters.
*   `/continue_train` (GET): continues a training session with specified parameters.
*   `/models` (POST): Lists models for a specific user.
*   `/upload` (POST): Uploads a trained model.
*   `/download` (GET): Downloads a trained model.
*   `/getRewards` (POST): Retrieves reward data for a specific model.
*   `/compareModels` (POST): Compares two trained models.
*   `/delete` (POST): Deletes a trained model.
*   `/test` (GET): Tests a trained model.
*   `/rename` (POST): Renames a trained model.
*   `/getModelSessions` (POST): Retrieves all sessions for a specific model.
*   `/getUserStats` (POST): Get user statistics.

## Contributing 🤝

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with descriptive messages.
4.  Submit a pull request.

## License 📜

This project has no specified license.

## Important Links 🔗

*   **Repository:** [https://github.com/pharoon/RoboGym](https://github.com/pharoon/RoboGym)

## Footer 📜

*   **Repository Name:** RoboGym
*   **Repository URL:** [https://github.com/pharoon/RoboGym](https://github.com/pharoon/RoboGym)
*   **Author:** [https://github.com/pharoon](https://github.com/pharoon)
*   **Contact:** example.com

⭐ Like this project? Give it a star!

🍴 Fork it to contribute and make it even better!

🐛 Find a bug? Report it!


---
**<p align="center">Generated by [ReadmeCodeGen](https://www.readmecodegen.com/)</p>**