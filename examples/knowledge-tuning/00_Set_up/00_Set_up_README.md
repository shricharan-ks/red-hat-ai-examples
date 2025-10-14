# Knowledge tuning example - set up

To get started, set up the working environment on your Red Hat OpenShift AI cluster and complete the example-specific configuration:
[Heading IDs](#heading-ids)

1. [Create an OpenShift project](#create-a-project)
2. [Create connections to storage](#create-connections-to-storage)
3. [Create a workbench](#create-a-workbench)
4. [Clone the example Git repository](#clone-the-example-git-repository)
5. [Set up your JupyterLab environment](#set-up-your-jupyterlab-environment)


## Create a project

To implement a data science workflow, you must create a project. Projects help your team to organize and work together on resources within separated namespaces. From a project you can create many workbenches, each with their own IDE environment (for example, JupyterLab), and each with their own connections and cluster storage. In addition, the workbenches can share models and data with pipelines and model servers.

### Prerequisites

* You have logged in to Red Hat OpenShift AI.

### Procedure

. On the navigation menu, select *Projects*. This page lists any existing projects that you have access to.

. Click *Create project*. 

. In the *Create project* modal, enter a display name and description.

. Click *Create*.

### Verification

You can see your project's initial state. To view more information about the project components and project access permissions, click a tab:

- **Workbenches** are instances of your development and experimentation environment. They typically contain individual development environments (IDEs), such as JupyterLab, RStudio, and Visual Studio Code.

- **Pipelines** contain the data science pipelines which run within the project.

- **Deployments** for quickly serving a trained model for real-time inference. You can have many model servers per data science project. One model server can host many models.

- **Cluster storage** is a persistent volume that retains the files and data you're working on within a workbench. A workbench has access to one or more cluster storage instances.

- **Connections** contain required configuration parameters for connecting to a data source, such as an S3 object bucket.

- **Permissions** define which users and groups can access the project.


## Create connections to storage

Add connections to your project for data inputs and object storage buckets. A connection is a resource that has the configuration parameters needed to connect to a data source or data sink, such as an AWS S3 object storage bucket.

For this example, you run a provided script that creates the following local Minio storage buckets for you:

- **My Storage** - Use this bucket for storing your models and data. You can reuse this bucket and its connection for your notebooks and model servers.

- **Pipelines Artifacts** - Use this bucket as storage for your pipeline artifacts. When you create a pipeline server, you need a pipeline artifacts bucket. For this example, create this bucket to separate it from the first storage bucket for clarity.

NOTE: Although you can use one storage bucket for both storing models and data and for storing pipeline artifacts, this example follows best practice and uses separate storage buckets for each purpose.

The provided script also creates a connection to each storage bucket. 

This example uses a disposable local Minio instance instead. The provided script automatically completes the following tasks for you: 

- Creates a Minio instance in your project.
- Creates two storage buckets in that Minio instance.
- Generates a random user id and password for your Minio instance.
- Creates two connections in your project, one for each bucket and both using the same credentials.
- Installs required network policies for service mesh functionality.

The [guide for deploying Minio](https://ai-on-openshift.io/tools-and-applications/minio/minio/) is the basis for this script.

IMPORTANT: The Minio-based Object Storage that the script creates is *not* meant for production usage.

### Prerequisites

You must know the OpenShift resource name for your data science project so that you run the provided script in the correct project. To get the project's resource name:

1. In the {productname-short} dashboard, select *Data science projects*.

2. Click the *?* icon next to the project name. 

A text box opens with information about the project, including its resource name:

### Procedure

1. In the {productname-short} dashboard, click the application launcher icon and then select the **OpenShift Console** option.

2. In the OpenShift console, click **+** in the top navigation bar.

3. Select your project from the list of projects.

4. Verify that you selected the correct project.

5. Copy the following code and paste it into the **Import YAML** editor.

NOTE: This code gets and applies the `setup-s3-no-sa.yaml` file.

```
apiVersion: v1
kind: ServiceAccount
metadata:
  name: demo-setup
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: demo-setup-edit
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: edit
subjects:
  - kind: ServiceAccount
    name: demo-setup
---
apiVersion: batch/v1
kind: Job
metadata:
  name: create-s3-storage
spec:
  selector: {}
  template:
    spec:
      containers:
        - args:
            - -ec
            - |-
              echo -n 'Setting up Minio instance and connections'
              oc apply -f https://github.com/rh-aiservices-bu/fraud-detection/raw/main/setup/setup-s3-no-sa.yaml
          command:
            - /bin/bash
          image: image-registry.openshift-image-registry.svc:5000/openshift/tools:latest
          imagePullPolicy: IfNotPresent
          name: create-s3-storage
      restartPolicy: Never
      serviceAccount: demo-setup
      serviceAccountName: demo-setup
```

6. Click **Create**.

### Verification

1. In the OpenShift console, there is a "Resources successfully created" message and a list of the following resources:

- `demo-setup`
- `demo-setup-edit`
- `create-s3-storage`

2.  In the Red Hat OpenShift AI dashboard:

a. Select **Data science projects** and then click the name of your project, **Fraud detection**.

b. Click **Connections**. There are two connections listed: `My Storage` and `Pipeline Artifacts`.

**IMPORTANT**
If your cluster uses self-signed certificates, your {productname-short} administrator might need to configure a certificate authority (CA) to securely connect to the S3 object storage, as described in [Accessing S3-compatible object storage with self-signed certificates(Self-Managed)](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/latest/html/installing_and_uninstalling_openshift_ai_self-managed/working-with-certificates_certs#accessing-s3-compatible-object-storage-with-self-signed-certificates_certs) or [Accessing S3-compatible object storage with self-signed certificates (Cloud Service)](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_cloud_service/1/html/installing_and_uninstalling_openshift_ai_cloud_service/working-with-certificates_certs#accessing-s3-compatible-object-storage-with-self-signed-certificates_certs).

## Create a workbench

A workbench is an instance of your development and experimentation environment. When you create a workbench, you select a workbench image that has the tools and libraries that you need for developing models. 

### Prerequisites

- You created a `My Storage` connection as described in the **Create connections to storage** section.

### Procedure

1. Navigate to the project detail page for the project that you created in **Create a project**.

2. Click the **Workbenches** tab, and then click **Create workbench**.

3. Fill out the name and description.

Red Hat OpenShift AI provides several supported workbench images. In the **Workbench image** section, you can select one of the default images or a custom image that an administrator has set up for you. The **Tensorflow** image has the libraries needed for this example.

4. Select the latest **Tensorflow** image.
<!-- edit for this example
-->

5. Choose a small deployment.

6. If your OpenShift cluster has available GPUs, the **Create workbench** form includes an **Accelerator** option. Select **None**. This example does not require any GPUs.
<!-- edit for this example
-->
7. Leave the default environment variables and storage options.

8. For **Connections**, click **Attach existing connection**.

9. Select `My Storage` (the object storage that you configured earlier) and then click **Attach**.

10. Click **Create workbench**.

### Verification

In the **Workbenches** tab for the project, the status of the workbench changes from `Starting` to `Running`.

NOTE: If you made a mistake, you can edit the workbench to make changes.

## Clone the example Git repository

The JupyterLab environment is a web-based environment, but everything you do inside it happens on Red Hat OpenShift AI and is powered by the OpenShift cluster. This means that, without having to install and maintain anything on your own computer, and without using valuable local resources such as CPU, GPU and RAM, you can conduct your work in this powerful and stable managed environment.

### Prerequisites

You created a workbench, as described in *Creating a workbench*.

### Procedure

1. Click the link for your workbench. If prompted, log in and allow JupyterLab to authorize your user.

Your JupyterLab environment window opens.

The file-browser window shows the files and folders that are saved inside your own personal space in OpenShift AI.

2. Bring the content of this example inside your JupyterLab environment:

a. On the toolbar, click the *Git Clone* icon:


b. Enter the following example Git *https* URL:

https://github.com/shricharan-ks/red-hat-ai-examples
<!-- edit for the URL
-->

c. Select the *Include submodules* option, and then click *Clone*.

d. In the file browser, double-click the newly-created *knowledge-tuning* folder.

e. In the left navigation bar, click the *Git* icon, and then click *Current Branch* to expand the branches and tags selector panel.

f.  On the *Branches* tab, in the *Filter* field, enter *{git-version}*.
<!-- should they use main for the git-version?
-->

g. Select *origin/{git-version}*. 

The current branch changes to *{git-version}*.


### Verification

In the file browser, view the notebooks that you cloned from Git.

## Set up your JupyterLab environment 

In JupyterLab, open the [00_Setup.ipynb](./00_Setup.ipynb) notebook and follow the instructions within the notebook.

### Next step

Congratulations! Your workbench is configured and ready for the knowledge training example. Throughout this example you will be guided through to a series of notebooks. Each notebook and supporting documentation provides  hands-on details about each step in the knowledge training pipeline.

Get started with [Data Preprocessing](./01_Data_Preprocessing/README.md)!



[def]: #Se