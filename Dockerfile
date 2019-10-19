FROM tensorflow/tensorflow:1.14.0-py3-jupyter

ENV MYDIR /notebooks/TextCNN
ENV LANG "C.UTF-8"
ENV LC_ALL "C.UTF-8"
ENV TZ "CST-8"

WORKDIR $MYDIR

COPY . $MYDIR/

RUN apt-get update && apt-get install -y graphviz
RUN cd $MYDIR && pip3 install -r requirements.txt -i http://pypi.douban.com/simple --trusted-host=pypi.douban.com

#COPY docker-entrypoint.sh /usr/local/bin/

EXPOSE 8000

#ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["/bin/bash"]

# docker build -t gswyhq/sentiment-analysis-textcnn -f Dockerfile .

