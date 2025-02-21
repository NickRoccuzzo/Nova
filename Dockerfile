# STEP 1: Use an official lightweight Python image
FROM python:3.11-slim

# STEP 2: Set the working directory inside the container
WORKDIR /app

# STEP 3: Copy all necessary application files into the container
COPY Nova.py NovaPyLogic.py tickers.json requirements.txt analysis.py ./

# STEP 4: Install dependencies (optimized)
RUN pip install --no-cache-dir -r requirements.txt

# STEP 5: Create a shared volume mount for storing output
VOLUME [ "/shared_data" ]

# STEP 6: Define an environment variable to control script execution
ENV RUN_ANALYSIS="false"

# STEP 7: Dynamically run either Nova.py or analysis.py
CMD ["sh", "-c", "if [ \"$RUN_ANALYSIS\" = \"true\" ]; then python analysis.py; else python Nova.py; fi"]
