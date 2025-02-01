# STEP 1: Use an official lightweight Python image
FROM python:3.11-slim

# STEP 2: Set the working directory inside the container
WORKDIR /app

# STEP 3: Copy application files into the container
COPY Nova.py NovaPyLogic.py tickers.json requirements.txt ./

# STEP 4: Install dependencies (using a clean and optimized method)
RUN pip install --no-cache-dir -r requirements.txt

# STEP 5: Create a shared volume mount for storing output
VOLUME [ "/shared_data" ]

# STEP 6: Define the environment variable (SECTOR) dynamically for each container
ENV SECTOR=""

# STEP 7: Run the main Python script when the container starts
CMD ["python", "Nova.py"]
