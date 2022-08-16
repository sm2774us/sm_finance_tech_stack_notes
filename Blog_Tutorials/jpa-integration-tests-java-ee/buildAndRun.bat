@echo off
call mvn clean package
call docker build -t de.rieckpil.blog/jpa-integration-tests-java-ee .
call docker rm -f jpa-integration-tests-java-ee
call docker run -d -p 8080:8080 -p 4848:4848 --name jpa-integration-tests-java-ee de.rieckpil.blog/jpa-integration-tests-java-ee