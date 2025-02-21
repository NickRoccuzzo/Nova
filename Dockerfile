# STEP 1: Use an official lightweight Python image
FROM python:3.11-slim

WORKDIR /app
COPY Nova.py NovaPyLogic.py tickers.json requirements.txt analysis.py ./

RUN pip install --no-cache-dir -r requirements.txt

# Install cron and add a cron job
RUN apt-get update && apt-get install -y cron && rm -rf /var/lib/apt/lists/*

# Add cron job to execute Nova.py every 15 minutes
RUN echo "*/15 * * * * root python /app/Nova.py" > /etc/cron.d/nova_cron \
    && chmod 0644 /etc/cron.d/nova_cron \
    && crontab /etc/cron.d/nova_cron

# Run cron in the foreground
CMD ["cron", "-f"]
