# Use Python 3.10 slim for maximum compatibility
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install -r requirements.txt

# Expose port for Render
EXPOSE 10000

# Start the app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
