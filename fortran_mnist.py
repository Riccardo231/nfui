import pathlib
import textwrap
import subprocess
import shutil
import os


def execute_fortran_mnist(config):
    """
    Generate Fortran source from config, write neural-fortran/example/cnn_mnist.f90,
    run cmake .. && make in neural-fortran/build, then execute ./cnn_mnist from a bin folder.
    Prints outputs and returns (src_path, fortran_source).
    """
    # --- helpers to convert layer configs to Fortran layer constructors ---
    def act_name(a):
        return (str(a) if a else "relu")

    def layer_to_fortran(l):
        t = l.get("type", "").lower()
        # compute activation call form once
        def act_call(a):
            if not a:
                return "relu()"
            return f"{a}()"

        if t == "dense":
            return f"dense({int(l.get('neurons', 32))}, {act_call(l.get('activation','relu'))})"
        if t == "conv2d":
            f = int(l.get("filters", l.get("neurons", 32)))
            kw = int(l.get("kernel_width", 3))
            kh = int(l.get("kernel_height", 3))
            return f"conv({f}, {kw}, {kh}, {act_call(l.get('activation','relu'))})"
        if t == "conv1d":
            f = int(l.get("filters", l.get("neurons", 32)))
            ks = int(l.get("kernel_size", 3))
            return f"conv({f}, {ks}, {act_call(l.get('activation','relu'))})"
        if t == "input":
            n = int(l.get("neurons", 0))
            return f"input({n})"
        if t == "dropout":
            rate = float(l.get("rate", l.get("dropout", 0.5)))
            return f"dropout({rate})"
        if t == "reshape":
            shape = l.get("shape", "")
            if isinstance(shape, (list, tuple)):
                nums = ", ".join(str(int(x)) for x in shape)
            else:
                nums = ", ".join(s.strip() for s in str(shape).split(",") if s.strip())
            return f"reshape({nums})"
        if t in ("locallyconnected", "locally_connected"):
            f = int(l.get("filters", l.get("neurons", 32)))
            ks = int(l.get("kernel_size", 3))
            return f"locallyconnected({f}, {ks}, {act_call(l.get('activation','relu'))})"
        if t == "maxpool1d":
            ps = int(l.get("pool_size", l.get("pool", 2)))
            st = int(l.get("stride", 1))
            return f"maxpool({ps}, {st})"
        if t == "maxpool2d":
            pw = int(l.get("pool_width", 2))
            ph = int(l.get("pool_height", 2))
            st = int(l.get("stride", 1))
            return f"maxpool({pw}, {ph}, {st})"
        if t == "flatten":
            return "flatten()"
        # fallback
        return f"dense({int(l.get('neurons', 32))}, {act_call(l.get('activation','relu'))})"

    # --- build the Fortran source text ---
    layers_cfg = config.get("layers", [])
    layer_exprs = [layer_to_fortran(l) for l in layers_cfg]
    joined = ", &\n        ".join(layer_exprs) if layer_exprs else "input(784), dense(128, relu)"

    use_list = sorted([
        "network", "sgd", "input", "conv", "maxpool", "flatten", "dense", "reshape",
        "dropout", "load_mnist","locally_connected", "label_digits", "softmax", "relu", "sigmoid", "tanhf"
    ])

    fortran_file = f"""
program mnist_network

use nf, only: {', '.join(use_list)}

implicit none

type(network) :: net

real, allocatable :: training_images(:,:), training_labels(:)
real, allocatable :: validation_images(:,:), validation_labels(:)
real, allocatable :: testing_images(:,:), testing_labels(:)
integer :: n
integer, parameter :: num_epochs = {int(config.get("epochs", 10))}

call load_mnist(training_images, training_labels, &
                validation_images, validation_labels, &
                testing_images, testing_labels)

! Construct network from configuration
net = network([ &
        {joined} &
    ])

call net%print_info()

epochs: do n = 1, num_epochs

    call net%train( &
        training_images, &
        label_digits(training_labels), &
        batch_size={int(config.get("batch_size", 10))}, &
        epochs=1, &
        optimizer={config.get("optimizer","sgd")}(learning_rate={float(config.get("learning_rate", 0.001))}) &
    )

    print '(a,i2,a,f5.2,a)', 'Epoch ', n, ' done, Accuracy: ', accuracy( &
        net, validation_images, label_digits(validation_labels)) * 100, ' %'

end do epochs

print '(a,f5.2,a)', 'Testing accuracy: ', &
    accuracy(net, testing_images, label_digits(testing_labels)) * 100, '%'

contains

real function accuracy(net, x, y)
    type(network), intent(in out) :: net
    real, intent(in) :: x(:,:), y(:,:)
    integer :: i, good
    good = 0
    do i = 1, size(x, dim=2)
        if (all(maxloc(net%predict(x(:,i))) == maxloc(y(:,i)))) then
            good = good + 1
        end if
    end do
    accuracy = real(good) / size(x, dim=2)
end function accuracy

end program mnist_network
"""
    fortran_file = textwrap.dedent(fortran_file).lstrip()

    # --- paths relative to this script ---
    base_dir = pathlib.Path(__file__).resolve().parent
    repo_dir = base_dir / "neural-fortran"
    example_dir = repo_dir / "example"

    # overwrite cnn_mnist.f90
    src_path = example_dir / "cnn_mnist.f90"
    src_path.write_text(fortran_file, encoding="utf-8")

    # build in repo_dir/build (fresh)
    build_dir = repo_dir / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    # run cmake .. && make
    try:
        subprocess.run(["cmake", ".."], cwd=str(build_dir), check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"cmake failed:\n{e.stdout}") from e

    try:
        subprocess.run(["make"], cwd=str(build_dir), check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"make failed:\n{e.stdout}") from e

    # prefer the build/bin location and run './cnn_mnist' with cwd set to that bin dir
    bin_dir = build_dir / "bin"
    exe_name = "cnn_mnist"
    exe_path = bin_dir / exe_name
    if not (bin_dir.exists() and exe_path.exists() and os.access(exe_path, os.X_OK)):
        # fallback: search for the executable anywhere in the repo and use its parent as cwd
        exe_path = None
        bin_dir = None
        for root, dirs, files in os.walk(repo_dir):
            if exe_name in files:
                candidate = pathlib.Path(root) / exe_name
                if candidate.exists() and os.access(candidate, os.X_OK):
                    exe_path = candidate
                    bin_dir = candidate.parent
                    break

    if not exe_path or not bin_dir:
        raise FileNotFoundError("Could not find built 'cnn_mnist' executable (expected at build/bin/cnn_mnist).")

    # run executable using './cnn_mnist' in the bin directory
    # start process and stream output line-by-line to the terminal (and capture)
    exec_output_lines = []
    try:
        popen = subprocess.Popen([f"./{exe_name}"], cwd=str(bin_dir),
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except Exception as e:
        raise RuntimeError(f"Failed to start executable '{exe_name}': {e}") from e

    # Read and print lines as they arrive, accumulate for return/error reporting
    try:
        with popen.stdout:
            for raw_line in iter(popen.stdout.readline, ""):
                if raw_line == "":
                    break
                line = raw_line.rstrip("\n")
                print(line, flush=True)                # show in terminal immediately
                exec_output_lines.append(line + "\n")  # accumulate with newline
        retcode = popen.wait()
    except Exception as e:
        popen.kill()
        raise

    exec_output = "".join(exec_output_lines)
    if retcode != 0:
        print("----- EXECUTABLE ERROR OUTPUT -----", flush=True)
        print(exec_output, flush=True)
        raise RuntimeError(f"Execution failed (exit {retcode}):\n{exec_output}")

    # print source path, source, and executable output to stdout (flush)
    try:
        print(f"Fortran source written to: {src_path}", flush=True)
        print("----- FORTRAN START -----", flush=True)
        print(fortran_file, flush=True)
        print("----- FORTRAN END -----", flush=True)
        print(f"Executable ran from: {exe_path}", flush=True)
        print("----- EXECUTABLE OUTPUT START -----", flush=True)
        print(exec_output, flush=True)
        print("----- EXECUTABLE OUTPUT END -----", flush=True)
    except Exception:
        pass

    # keep previous return shape for UI compatibility
    return src_path, fortran_file




