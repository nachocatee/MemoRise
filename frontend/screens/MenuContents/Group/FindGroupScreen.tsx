import { useNavigation } from "@react-navigation/native";
import { StackNavigationProp } from "@react-navigation/stack";
import axios from "axios";
import { useEffect, useState } from "react";
import {
  View,
  Text,
  FlatList,
  Pressable,
  Image,
  Modal,
  Alert,
} from "react-native";
import LinearGradient from "react-native-linear-gradient";
import { useSelector } from "react-redux";

import SmallBtn from "../../../components/Button/SmallBtn";
import GoBackHeader from "../../../components/Header/GoBackHeader";
import SeachInput from "../../../components/SearchInput";
import { calculateDynamicWidth } from "../../../constants/dynamicSize";
import { RootState } from "../../../store/store";
import { BACKEND_URL } from "../../../util/http";
import { styles } from "./GroupStyle";
import GroupBox from "../../../components/GroupBox";
import AlertModal from "../../../components/Modal/AlertModal";
import { BlurView } from "@react-native-community/blur";
import PasswordInputModal from "../../../components/Modal/Group/PasswordInputModal";

type RootStackParamList = {
  MakeGroup: undefined;
  GroupDetail: GroupDetailParams;
  GroupSearchResult: { searchResults: GroupData };
};

export type GroupDetailParams = {
  teamSeq: number;
  userSeq: number;
};

export type GroupData = [
  {
    teamSeq: number;
    teamName: string;
    profiles: [string];
    password: boolean;
    participated: boolean;
  }
];

const FindGroupScreen = () => {
  const navigation = useNavigation<StackNavigationProp<RootStackParamList>>();
  const MakeGroupHandler = () => navigation.navigate("MakeGroup");
  const userId = useSelector((state: RootState) => state.userInfo.id);
  const [groupData, setGroupData] = useState<GroupData | null>(null);

  // 그룹 검색
  const [searchResultList, setSearchResultList] = useState<GroupData | null>(
    null
  );
  const [searchKeyword, setSearchKeyword] = useState("");

  const searchInputHandler = (enteredText: string) => {
    setSearchKeyword(enteredText);
  };

  const getSearchGroupList = async () => {
    if (searchKeyword) {
      try {
        const res = await axios({
          method: "GET",
          // url: BACKEND_URL + `/teams/${userId}`,
          url: BACKEND_URL + `/teams/30`, // 더미 데이터
          params: {
            keyword: searchKeyword,
          },
        });
        console.log(res.data);
        setSearchResultList(res.data);
        setSearchKeyword("");
        navigation.navigate("GroupSearchResult", { searchResults: res.data });
      } catch (err) {
        console.log(err);
      }
    } else {
      Alert.alert("검색어를 입력해 주세요!");
    }
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

  // 비밀번호 입력
  const [groupPassword, setGroupPassword] = useState("");
  const [enteredPassword, setEnteredPassword] = useState("");
  const [invalid, setInvalid] = useState(false);

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
        // userSeq: userId,
        userSeq: 30, // 더미 데이터
        password: password,
      },
    })
      .then((res) => {
        console.log(res);
        if (res.data.message === "참여 코드가 일치하지 않습니다.") {
          Alert.alert("비밀번호가 일치하지 않습니다.");
        } else {
          // 디테일로
          navigation.navigate("GroupDetail", {
            teamSeq: teamSeq,
            // userSeq: userId,
            userSeq: 30, // 더미 데이터
          });
        }
      })
      .catch((err) => {
        console.log(err);
      });
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
        // userSeq: userId,
        userSeq: 30, // 더미 데이터
      });
    } else {
      setTargetTeamName(teamName);
      openAlertModal(teamSeq, password);
    }
  };

  // 전체 그룹 목록 가져오기
  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await axios({
          method: "GET",
          // url: BACKEND_URL + `/teams/${userId}`,
          url: BACKEND_URL + `/teams/30`, // 더미 데이터
        });
        setGroupData(res.data);
      } catch (err) {
        console.log(err);
      }
    };
    fetchData();
  }, []);

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
          <SeachInput
            onChangeText={searchInputHandler}
            value={searchKeyword}
            onSubmitEditing={getSearchGroupList}
          />
        </View>
        {groupData && (
          <FlatList
            style={{ flex: 1 }}
            contentContainerStyle={styles.groupContainer}
            data={groupData}
            keyExtractor={(item) => item.teamSeq.toString()}
            renderItem={({ item }) => (
              <View>
                <GroupBox
                  teamName={item.teamName}
                  myProfile={item.profiles[0]}
                  memberProfiles={item.profiles}
                  ownerProfile={null}
                  owner={false}
                  goDetailHandler={() =>
                    onPressGroupHandler(
                      item.teamSeq,
                      item.participated,
                      item.password,
                      item.teamName
                    )
                  }
                  teamSeq={item.teamSeq}
                />
                {item.password && (
                  <View style={styles.lockIconContainer}>
                    <Image
                      style={styles.lockIcon}
                      source={require("../../../assets/image/lock.png")}
                    />
                  </View>
                )}
              </View>
            )}
          />
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
                invalid={invalid}
              />
            </View>
          </Modal>
        </BlurView>
      )}
    </LinearGradient>
  );
};

export default FindGroupScreen;
