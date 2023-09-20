import React, { useState } from "react";
import { View, Text, Modal, Pressable, Image } from "react-native";
import { BlurView } from "@react-native-community/blur";

import MainHeader from "../../components/Header/MainHeader";
import { styles } from "./MainStyle";

const MainScreen = () => {
  // 알림 모달
  const [isNotificationModalVisible, setNotificationModalVisible] =
    useState(false);

  const closeNotificationModal = () => {
    setNotificationModalVisible(false);
  };

  // 추가 버튼 모달
  const [isMemoBtnModalVisible, setMemoBtnModalVisible] = useState(false);

  const closeMemoBtnModal = () => {
    setMemoBtnModalVisible(false);
  };

  return (
    <View style={{ flex: 1 }}>
      {!isNotificationModalVisible && (
        <View style={styles.headerContainer}>
          <MainHeader openModal={() => setNotificationModalVisible(true)} />
        </View>
      )}
      <View style={styles.rootContainer}>
        <Pressable
          style={styles.btnContainer}
          onPress={() => setMemoBtnModalVisible(true)}
        >
          <Image
            source={require("../../assets/image/mainbtn.png")}
            style={styles.addBtn}
          />
        </Pressable>
      </View>

      {/* 알림 모달 */}
      {isNotificationModalVisible && (
        <BlurView
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
          }}
          blurType="dark"
          blurAmount={4}
        >
          <Modal
            transparent={true}
            animationType="slide"
            visible={isNotificationModalVisible}
            onRequestClose={closeNotificationModal}
          >
            {/* 헤더 */}
            <View style={styles.header}>
              <Pressable>
                <Image
                  source={require("../../assets/image/logo/logowhite.png")}
                  style={styles.logo}
                />
              </Pressable>
              <Pressable
                style={styles.cancelContainer}
                onPress={closeNotificationModal}
              >
                <Image
                  source={require("../../assets/icons/cancelwhite.png")}
                  style={styles.cancel}
                />
              </Pressable>
            </View>

            <View style={styles.modalEmptyContainer}>
              <Text style={styles.modalEmpty}>알림 없음</Text>
            </View>
          </Modal>
        </BlurView>
      )}

      {/* 메모 버튼 모달 */}
      {isMemoBtnModalVisible && (
        <BlurView
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
          }}
          blurType="dark"
          blurAmount={4}
        >
          <Modal
            transparent={true}
            animationType="fade"
            visible={isMemoBtnModalVisible}
            onRequestClose={closeMemoBtnModal}
          >
            <View style={styles.memoBtnWrap}>
              <View style={styles.memoBtnContainer}>
                <Text style={styles.memoBtnText}>메모 조회하기</Text>
                <Pressable>
                  <Image
                    source={require("../../assets/image/memoreadbtn.png")}
                    style={styles.addBtn}
                  />
                </Pressable>
              </View>
              <View style={styles.memoBtnContainer}>
                <Text style={styles.memoBtnText}>메모 작성하기</Text>
                <Pressable>
                  <Image
                    source={require("../../assets/image/memocreatebtn.png")}
                    style={styles.addBtn}
                  />
                </Pressable>
              </View>
            </View>

            <Pressable onPress={closeMemoBtnModal} style={styles.btnContainer}>
              <Image
                source={require("../../assets/image/cancelbtn.png")}
                style={styles.addBtn}
              />
            </Pressable>
          </Modal>
        </BlurView>
      )}
    </View>
  );
};

export default MainScreen;
