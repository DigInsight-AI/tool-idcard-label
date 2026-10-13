import argparse
import webbrowser

from .server import ReviewServer
from .store import ReviewStore


def build_parser():
    parser = argparse.ArgumentParser(description="Create, review, and correct ID-card pose annotations.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Start the local review web app.")
    serve.add_argument("--input-root", help="Image directory for an explicit startup workspace.")
    serve.add_argument("--workspace", help="Optional startup workspace; otherwise choose directories in the app.")
    serve.add_argument("--limit", type=int, help="Queue at most N images for testing.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8765, type=int)
    serve.add_argument("--no-browser", action="store_true")
    serve.add_argument("--refresh-confidence", default=0.25, type=float)
    serve.add_argument("--refresh-keypoint-confidence", default=0.5, type=float)
    serve.add_argument("--refresh-device")
    serve.add_argument(
        "--workspace-class-id",
        type=int,
        choices=(0, 1),
        help="Normalize this workspace's stored annotations to class 0 or 1.",
    )

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    store = None
    if args.workspace:
        store = ReviewStore(args.workspace, args.workspace_class_id)
        if args.input_root:
            result = store.initialize(args.input_root, args.limit)
            print("Queued %d images (%d unreadable images)." % (result["imported"], len(result["errors"])))
        elif not store.meta("input_root"):
            store.close()
            raise SystemExit("--input-root is required when creating an explicit workspace")
    server = ReviewServer(
        (args.host, args.port),
        store,
        workspace_class_id=args.workspace_class_id,
        refresh_confidence=args.refresh_confidence,
        refresh_keypoint_confidence=args.refresh_keypoint_confidence,
        refresh_device=args.refresh_device,
    )
    url = "http://%s:%d" % (args.host, args.port)
    print("Review tool: %s" % url)
    if server.store:
        print("Workspace: %s" % server.store.workspace)
    else:
        print("Choose input and output directories in the browser.")
    if server.model_refresher:
        print("Refresh model: %s" % server.model_refresher.describe()["selected_path"])
    else:
        print("Refresh model: configure a model directory in Settings.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping review tool.")
    finally:
        server.server_close()
        server.close_store()
