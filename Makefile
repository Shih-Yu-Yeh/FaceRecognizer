include $(TOPDIR)/rules.mk

PKG_NAME:=recognizer
PKG_RELEASE:=1

include $(INCLUDE_DIR)/kernel.mk
include $(INCLUDE_DIR)/package.mk
include $(INCLUDE_DIR)/cmake.mk

define Package/recognizer
  SECTION:=xs-ai
  CATEGORY:=xs-ai
  TITLE:=Face Recognizer Example
  DEPENDS:= +opencv +cnpy +libstdcpp +libpthread
endef

define Build/Prepare
	mkdir -p $(PKG_BUILD_DIR)
	$(CP) ./src/* $(PKG_BUILD_DIR)
endef

define Package/recognizer/description
	Face Recognizer Example
endef

define Package/recognizer/install
	$(INSTALL_DIR) $(1)/bin
	$(INSTALL_BIN) $(PKG_BUILD_DIR)/$(PKG_NAME) $(1)/bin
endef

$(eval $(call BuildPackage,recognizer))

