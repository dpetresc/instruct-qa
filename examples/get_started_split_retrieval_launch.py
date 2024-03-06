import sys
import pkgutil
import get_started_split_retrieval_bis
import importlib
import instruct_qa
import traceback

loading_cached = None

def reload_package(package):
    for importer, modname, ispkg in pkgutil.walk_packages(path=package.__path__, prefix=package.__name__ + '.'):
        if not ispkg:
            module = sys.modules.get(modname)
            if module:
                importlib.reload(module)

if __name__ == "__main__":
    while True:
        try:
            if not loading_cached:
                loading_cached = get_started_finegrain.loading()

            get_started_finegrain.running(loading_cached)
            print("Press enter to re-run the script, CTRL-C to exit")
            sys.stdin.readline()

            # Reload all modules in instruct_qa
            reload_package(instruct_qa)
            # Reload get_started_finegrain after reloading all instruct_qa modules
            importlib.reload(get_started_finegrain)

        except KeyboardInterrupt:
            print("\nExiting the program.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
            continue
        
