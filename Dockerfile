###############################################################################
# Stage 1: Builder ¡ª install Python deps and download vendor assets
###############################################################################
FROM python:3.11-slim AS builder
ADD deb-sources.list /etc/apt/sources.list.d/debian.sources

ARG PRODUCTION=0
ARG LIGHTWEIGHT=0

WORKDIR /app

# gcc is needed to compile C extensions during pip install
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt requirements-embeddings.txt constraints.txt ./
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir --prefix=/install -c constraints.txt -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    if [ "$LIGHTWEIGHT" = "0" ]; then \
        pip install --no-cache-dir --prefix=/install -c constraints.txt -r requirements-embeddings.txt -i https://pypi.tuna.tsinghua.edu.cn/simple; \
    fi

# Download vendor assets (JS/CSS/fonts)
RUN mkdir -p /app/static/vendor
COPY scripts/download_offline_deps.py scripts/
RUN pip install --no-cache-dir requests -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    PRODUCTION=${PRODUCTION} python scripts/download_offline_deps.py && \
    echo "? Vendor dependencies downloaded successfully"

###############################################################################
# Stage 2: FFmpeg ¡ª download static binaries (much smaller than apt ffmpeg)
###############################################################################
FROM python:3.11-slim AS ffmpeg-stage
ADD deb-sources.list /etc/apt/sources.list.d/debian.sources

RUN apt-get update && apt-get install -y --no-install-recommends wget xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && ARCH=$(dpkg --print-architecture) \
    && wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-${ARCH}-static.tar.xz -O /tmp/ff.tar.xz \
    && mkdir -p /tmp/ffmpeg-dir \
    && tar xf /tmp/ff.tar.xz -C /tmp/ffmpeg-dir --strip-components=1 \
    && mv /tmp/ffmpeg-dir/ffmpeg /usr/local/bin/ffmpeg \
    && mv /tmp/ffmpeg-dir/ffprobe /usr/local/bin/ffprobe \
    && chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe \
    && rm -rf /tmp/ff.tar.xz /tmp/ffmpeg-dir

###############################################################################
# Stage 3: Runtime ¡ª lean final image with only what's needed
###############################################################################
FROM python:3.11-slim
ADD deb-sources.list /etc/apt/sources.list.d/debian.sources

WORKDIR /app

# Copy static ffmpeg binaries (~150MB vs ~450MB from apt)
COPY --from=ffmpeg-stage /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg
COPY --from=ffmpeg-stage /usr/local/bin/ffprobe /usr/local/bin/ffprobe

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy downloaded vendor assets from builder
COPY --from=builder /app/static/vendor /app/static/vendor

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /data/uploads /data/instance && chmod 755 /data/uploads /data/instance

# Set environment variables
ENV FLASK_APP=src/app.py
ENV SQLALCHEMY_DATABASE_URI=sqlite:////data/instance/transcriptions.db
ENV UPLOAD_FOLDER=/data/uploads
ENV PYTHONPATH=/app
ENV HF_HOME=/data/instance/huggingface

# Add entrypoint script
COPY scripts/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8899

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["gunicorn", "--workers", "3", "--bind", "0.0.0.0:8899", "--timeout", "600", "src.app:app"]
