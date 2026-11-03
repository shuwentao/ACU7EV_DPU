APP = drm-dual-plane

# Add any other object files to this list below
APP_OBJS = drm-dual-plane.o

all: build

build: $(APP)

$(APP): $(APP_OBJS)
	$(CC) -o $@ $(APP_OBJS) $(LDFLAGS) $(LDLIBS)
clean:
	rm -f $(APP) *.o

