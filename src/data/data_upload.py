import paramiko
import os


def upload_folder_to_hpc(local_folder, remote_folder, hostname, port, username, password):
    try:
        # Create an SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Connect to the HPC environment
        ssh.connect(hostname, port, username, password)
        
        # Create an SFTP session
        sftp = ssh.open_sftp()
        
        # Walk through the local folder
        for root, dirs, files in os.walk(local_folder):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_folder)
                remote_path = os.path.join(remote_folder, relative_path)
                
                # Create remote directories if they don't exist
                remote_dir = os.path.dirname(remote_path)
                try:
                    sftp.stat(remote_dir)
                except FileNotFoundError:
                    sftp.mkdir(remote_dir)
                
                # Upload the file
                sftp.put(local_path, remote_path)
        
        # Close the SFTP session and SSH connection
        sftp.close()
        ssh.close()
        
        print("Folder uploaded successfully.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
local_folder = '/path/to/local/folder'
remote_folder = '/path/to/remote/folder'
hostname = 'hpc.example.com'
port = 22
username = 'your_username'
password = 'your_password'

upload_folder_to_hpc(local_folder, remote_folder, hostname, port, username, password)