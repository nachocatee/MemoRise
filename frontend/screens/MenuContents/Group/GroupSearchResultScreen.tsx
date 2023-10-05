import React, { useState } from "react";
import { View, Text, Alert, Image, ScrollView, Modal } from "react-native";
import LinearGradient from "react-native-linear-gradient";
import { RouteProp, NavigationProp } from "@react-navigation/native";
import { useSelector } from "react-redux";
import { BlurView } from "@react-native-community/blur";
import axios from "axios";

import GoBackHeader from "../../../components/Header/GoBackHeader";
import { GroupData, GroupDetailParams } from "./FindGroupScreen";
import { styles } from "./GroupStyle";
import SmallBtn from "../../../components/Button/SmallBtn";
import { calculateDynamicWidth } from "../../../constants/dynamicSize";
import SearchInput from "../../../components/SearchInput";
import { BACKEND_URL } from "../../../util/http";
import { RootState } from "../../../store/store";
import GroupBox from "../../../components/GroupBox";
import AlertModal from "../../../components/Modal/AlertModal";
import PasswordInputModal from "../../../components/Modal/Group/PasswordInputModal";

type RootStackParamList = {
  MakeGroup: undefined;
  GroupDetail: GroupDetailParams;
  GroupSearchResult: { searchResults: GroupData };
};

type GroupSearchResultScreenRouteProp = RouteProp<
  RootStackParamList,
  "GroupSearchResult"
>;

type GroupSearchResultScreenNavigationProp = NavigationProp<
  RootStackParamList,
  "GroupSearchResult"
>;

type Props = {
  route?: GroupSearchResultScreenRouteProp;
  navigation?: GroupSearchResultScreenNavigationProp;
};

const GroupSearchResultScreen: React.FC<Props> = ({ route, navigation }) => {
  if (!route || !navigation) return null;
  const userId = useSelector((state: RootState) => state.userInfo.id);

  const searchResults = route.params.searchResults;

  const MakeGroupHandler = () => navigation.navigate("MakeGroup");

  // 비밀번호 입력
  const [groupPassword, setGroupPassword] = useState("");

  const passwordModalHandler = () => {
    setPasswordModalVisible(true);
    setGroupPassword("");
  };

  const passwordInputHandler = (enteredText: string) => {
    setGroupPassword(enteredText);
  };

  const passwordConfirmHandler = () => {
    joinGroupHandler(targetTeamSeq, groupPassword);
  };

  // 비밀번호 입력 모달
  const [isPasswordModalVisible, setPasswordModalVisible] = useState(false);

  const closePasswordModal = () => {
    setPasswordModalVisible(false);
  };

  // 그룹 참여 axios
  const joinGroupHandler = (teamSeq: number, password: string) => {
    axios({
      method: "POST",
      url: BACKEND_URL + `/teams/${teamSeq}`,
      data: {
        userSeq: userId,
        password: password,
      },
    })
      .then((res) => {
        if (res.data.message === "참여 코드가 일치하지 않습니다.") {
          Alert.alert("비밀번호가 일치하지 않습니다.");
        } else {
          // 디테일로
          navigation.navigate("GroupDetail", {
            teamSeq: teamSeq,
            userSeq: userId,
          });
        }
      })
      .catch((err) => {
        console.log(err);
      });
  };

  // 그룹 참가 확인 모달
  const [isAlertModalVisible, setAlertModalVisible] = useState(false);
  const [targetTeamSeq, setTargetTeamSeq] = useState(0);
  const [targetTeamPassword, setTargetTeamPassword] = useState(false);
  const [targetTeamName, setTargetTeamName] = useState("");

  const openAlertModal = (teamSeq: number, password: boolean) => {
    setAlertModalVisible(true);
    setTargetTeamSeq(teamSeq);
    setTargetTeamPassword(password);
  };

  // 취소 버튼 눌렀을 때
  const closeAlertModal = () => {
    setAlertModalVisible(false);
  };

  // 확인 버튼 눌렀을 때
  const joinConfirmHandler = () => {
    if (targetTeamPassword) {
      // 비공개 그룹
      passwordModalHandler();
    } else {
      // 공개 그룹
      joinGroupHandler(targetTeamSeq, "");
    }
  };

  // 그룹 터치 시
  const onPressGroupHandler = (
    teamSeq: number,
    participated: boolean,
    password: boolean,
    teamName: string
  ) => {
    // 이미 참여 중인 그룹일 때
    if (participated) {
      navigation.navigate("GroupDetail", {
        teamSeq: teamSeq,
        userSeq: userId,
      });
    } else {
      setTargetTeamName(teamName);
      openAlertModal(teamSeq, password);
    }
  };

  // 그룹 검색
  const [searchKeyword, setSearchKeyword] = useState("");
  const [searchResultList, setSearchResultList] = useState<GroupData | null>(
    searchResults
  );

  const searchInputHandler = (enteredText: string) => {
    setSearchKeyword(enteredText);
  };

  const getSearchGroupList = async () => {
    if (searchKeyword) {
      try {
        const res = await axios({
          method: "GET",
          url: BACKEND_URL + `/teams/${userId}`,
          params: {
            keyword: searchKeyword,
          },
        });
        console.log(res.data);
        setSearchKeyword("");
        setSearchResultList(res.data);
      } catch (err) {
        console.log(err);
      }
    } else {
      Alert.alert("검색어를 입력해 주세요!");
    }
  };

  console.log(searchResultList);

  return (
    <LinearGradient
      colors={["#F5F5F5", "#E9E9E9"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 0, y: 1 }}
      style={{ flex: 1 }}
    >
      <GoBackHeader />
      <View style={styles.container}>
        <View style={styles.titleContainer}>
          <Text style={styles.title}>그룹 찾기</Text>
          <SmallBtn onPress={MakeGroupHandler}>만들기</SmallBtn>
        </View>
        <View style={{ marginTop: calculateDynamicWidth(25) }}>
          <SearchInput
            onChangeText={searchInputHandler}
            value={searchKeyword}
            onSubmitEditing={getSearchGroupList}
          />
        </View>
        {searchResultList && searchResultList[0] ? (
          <ScrollView
            style={{ flex: 1 }}
            contentContainerStyle={styles.groupContainer}
          >
            {searchResultList.map((group, idx) => (
              <View key={idx}>
                <GroupBox
                  teamName={group.teamName}
                  myProfile={group.profiles[0]}
                  memberProfiles={group.profiles}
                  ownerProfile={null}
                  owner={false}
                  goDetailHandler={() =>
                    onPressGroupHandler(
                      group.teamSeq,
                      group.participated,
                      group.password,
                      group.teamName
                    )
                  }
                  teamSeq={group.teamSeq}
                />
                {group.password && (
                  <View style={styles.lockIconContainer}>
                    <Image
                      style={styles.lockIcon}
                      source={require("../../../assets/image/lock.png")}
                    />
                  </View>
                )}
              </View>
            ))}
          </ScrollView>
        ) : (
          <View style={styles.emptyResultContainer}>
            <Image
              style={styles.group}
              source={require("../../../assets/icons/group.png")}
            />
            <Text style={styles.empty}>검색 결과가 없습니다</Text>
          </View>
        )}
      </View>

      {/* 그룹 참가 확인 모달 */}
      {isAlertModalVisible && (
        <BlurView
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
          }}
          blurType="light"
          blurAmount={4}
        >
          <AlertModal
            modalVisible={isAlertModalVisible}
            closeModal={closeAlertModal}
            onConfirm={joinConfirmHandler}
            contentText={`${targetTeamName} 에\n참가하시겠습니까?`}
            btnText="확인"
          />
        </BlurView>
      )}

      {/* 비밀번호 모달 */}
      {isPasswordModalVisible && (
        <BlurView
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
          }}
          blurType="light"
          blurAmount={4}
        >
          <Modal
            transparent={true}
            animationType="fade"
            visible={isPasswordModalVisible}
            onRequestClose={closePasswordModal}
          >
            <View style={styles.modalContainer}>
              <PasswordInputModal
                closeModal={closePasswordModal}
                onChangeText={passwordInputHandler}
                value={groupPassword}
                onConfirm={passwordConfirmHandler}
                invalid={false}
              />
            </View>
          </Modal>
        </BlurView>
      )}
    </LinearGradient>
  );
};

export default GroupSearchResultScreen;
