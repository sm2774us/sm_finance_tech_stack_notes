what's this Kubernetes all about? Well Kubernetes is one of several cluster operating systems you can choose from at the moment, right? And it's definitely the most popular.

Maybe you've heard of DCOS. Maybe you've heard of Rancher. Maybe you've heard of Openshift if you're in a commercial environment which is basically a commercialized multi-tenant version of Kubernetes. Maybe you have heard of Giant Swarm which is a managed Kubernetes service in the cloud, or Nomad or Docker Datacenter. But Kubernetes is definitely the technology where the most attention is at the moment. So this is the basic architecture of a Kubernetes cluster. On the left hand side you'll see the Kubernetes master, okay? You may have one master, you may have several masters for high availability setups.


![](https://d33wubrfki0l68.cloudfront.net/518e18713c865fe67a5f23fc64260806d72b38f5/61d75/images/docs/post-ccm-arch.png)

And you as an administrator you mainly interact with the master via a rest API or later on by the Kubernetes CLI. So this master offers an API server, it offers a control manager, the scheduler which is responsible for scheduling tasks and containers to the minions, and it also contains an ETCD value store which basically is the brain of the whole Kubernetes cluster.

So this ETCD holds the whole state of this cluster. On the right hand side you have these so called minions, or sometimes called worker nodes, okay? So as a user when you access certain services that run on the Kubernetes cluster you usually interact with those minions. Now, in the minion you have a kube-proxy which is responsible for taking inbound traffic and directing it to the containers running within it, you have a kubelet well, which is responsible for managing the container engine and for managing the ports and containers that run within this container engine and it's the master node that talks to well the minions via the kubelet.

So at the heart of the minion you have a container engine. Well usually there's the container engine docker, but you could also use rocket as a container engine. And the container engine is then responsible for running the pods and the containers. So here are some key concepts and building blocks you should definitely know when working with Kubernetes, okay? Well we talked about the pods in the previous slides so let's start with them.

So pods are the smallest deployable compute unit in Kubernetes, okay? So think of pods as a grouping of well, docking containers maybe, right? So a pod can contain one or many docker containers. Now the important thing to know is that if you have multiple containers in a pod, okay? They share a same fate, okay? So if one of the containers dies or exits, the whole pod would exit and all of the contained containers.

So if you think about well how you can divide up application containers into pods, well you have to make the choice if you stick one container in one pod, or maybe multiple containers in one pod, now those pods they can be given labels, okay? So labels are basically arbitrary key value pairs used to identify objects within Kubernetes. So later on we can then say show me all the pods with label back N two, or show me all the pods that are in the test environment, and so on and so forth.

So use labels to identify things within Kubernetes. At the very top you see a service. And the service is an important concept in Kubernetes. Well pods are dynamic, okay? Pods have a unique IP address. Pods come and go. So as a client for those services or those containers within the pods, you never know which one to talk to, and this is what the service if for.

It's an extraction for a logical grouping of pods, okay? So the service has a unique IP address and it has a DNS name which you can use to talk to the pods behind the service. And it's the service responsibility to, well to proxy any inbound calls to the running pods behind it. But as a client, you're not aware of the number of pods behind a service, okay? Can be one, but it could be many.

Now the replica sets insure that always the required of number of pod replicas are running within your Kubernetes cluster, okay? So if you say I always want three instances of this pod running, it's the replica set that takes care of well managing this. So if one pod dies, the replica set will spin up another one so you're up to the desired amount of replicas again. Or if you tell the replica set hey, I need two more, then it's the replica set that will spin up the two more instances.

Now last but not least it's the deployment you use to declare pods and replica controllers and labels and volumes, all in one go. So the deployment is a major concept, a major building block which we'll be using throughout this section and throughout these videos. Now just briefly, how can you setup Kubernetes in the cloud or locally? Well, technically you can run Kubernetes almost anywhere, right? Though if you download the Kubernetes distribution, there's a shell script provided and what you can do is you can specify a few additional environment variables like you see here.


For example to have GCE, so that's the Google Computer Engine as a Kubernetes provider, or maybe AWS as an infrastructure, as a service provider to host your Kubernetes cluster. So just export a few environment variables and then issue the curl request you see here. And after maybe 30 minutes you have a running Kubernetes cluster in the cloud. Now obviously throughout this course we want to have, well short development round trips, so what we use is we use something called Minikube instead, okay? So Minikube lets you run a very small Kubernetes cluster locally on your machine.


So this is what we will be using through this section and throughout those following videos. Now, the main tool we use to interact with out Kubernetes clusters is the Kubernetes command line interface. And I'm going to show you this live. So let's open our console, I issue Minikube status and we should see the status of our Minikube server, shortly.
So the Minikube VM is running and the localkube is running as well. So kube control is the command line interface we use to interact with out Kubernetes cluster, so you see here we have a lot of commands available. And maybe the most basic command is kube control cluster info. So this tells us some basic information about our Kubernetes cluster.


So this is the Kubernetes master running here on a local IP address, the Kube DNS service and the Kubernetes dashboard. So let's have a look at the dashboard. So do Minikube dashboard and now a browser should open up in our default browser and you see here hopefully in a few seconds that the dashboard is coming up, okay? So if you don't like interacting with Kubernetes in the command line you can always use this dashboard instead, okay? So what we can for example do is kube control get nodes to display the number of nodes, and basically we only expect one node here because we're running locally.

And just to show you you can say group kube control config and now we see here those are the available contexts. I have Minikube currently. Now here I switched to this context and what I do now is I use kube control and get nodes, and you see here I see three nodes.

This is a Kubernetes cluster instance running in the cloud. Okay so this was a very basic introduction to Kubernetes and its key building blocks and in the next video we're going to deploy our Go microservice to our locally running Kubernetes. Hope to see you then, bye-bye.
 Language: English   About  Becom